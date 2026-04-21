import queue
import threading
import time
from functools import partial
from typing import Dict, List, Optional

import yaml

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.tracker import (
    BaseTracker,
    _CameraInfoCollector,
    _spin_ros_node,
)
from cuvslam_examples.realsense.utils import Landmark, Pose


# ---------- Local helpers for raw-image OAK stereo (no realsense dep) ----------

def _oak_image_to_raw_camera_info_topic(image_topic: str) -> str:
    """OAK raw convention: /oak_base_front/left/image_raw -> /oak_base_front/left/camera_info"""
    parent = image_topic.rsplit("/", 1)[0]
    return parent + "/camera_info"


def _oak_extrinsic_from_tf(left_frame: str, right_frame: str, timeout_s: float = 10.0):
    """Return 4x4 pose of right-camera-frame IN left-camera-frame via /tf.

    Used as rig_from_camera for the right camera when the left camera is the rig origin.
    """
    import numpy as np
    import rclpy
    import tf2_ros
    from rclpy.node import Node
    from scipy.spatial.transform import Rotation

    node = Node("oak_stereo_extrinsic_tf")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)

    deadline = time.time() + timeout_s
    msg = None
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        try:
            msg = buf.lookup_transform(left_frame, right_frame, rclpy.time.Time())
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue
    node.destroy_node()
    if msg is None:
        raise TimeoutError(f"tf lookup {left_frame} -> {right_frame} timed out after {timeout_s}s")

    t = msg.transform.translation
    q = msg.transform.rotation
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def _make_oak_raw_camera(k, d, width: int, height: int, rig_from_camera_4x4) -> vslam.Camera:
    """Build a cuvslam.Camera from raw K, D (ROS rational_polynomial), image size,
    and the 4x4 pose of this camera in the rig frame.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    cam = vslam.Camera()
    # K is a flat 9-list from CameraInfo.k
    cam.focal = (k[0], k[4])
    cam.principal = (k[2], k[5])
    cam.size = (width, height)
    # cuvslam Polynomial = OpenCV rational polynomial, exactly 8 coeffs:
    # [k1, k2, p1, p2, k3, k4, k5, k6]. ROS rational_polynomial D has 14
    # elements; the first 8 are this set (indices 8..13 are thin-prism /
    # tilted terms that cuvslam's Polynomial doesn't model).
    d8 = [float(d[i]) if i < len(d) else 0.0 for i in range(8)]
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, d8)

    m = np.asarray(rig_from_camera_4x4, dtype=np.float64)
    cam.rig_from_camera = vslam.Pose(
        rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
        translation=m[:3, 3],
    )
    return cam

DEFAULT_OAK_RIG_FILE = "oak_rig.yaml"

SLOP_SEC = 0.001


def _oak_image_to_compressed_topic(image_topic: str) -> str:
    """OAK images are recorded as compressed: /oak_base_front/left/image_raw/compressed"""
    return image_topic + "/compressed"


def _decode_oak_compressed_image(msg) -> "np.ndarray | None":
    import cv2
    import numpy as np

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return np.ascontiguousarray(img) if img is not None else None


class RosOakMulticamTracker(BaseTracker):
    """Multi-camera stereo tracking from N OAK-D stereo pairs via ROS topics.

    Rig schema (oak_rig.yaml): per camera only `serial_no` + L/R rig transforms.
    Everything else (key, robot_part, image_mode, topics, rect_calib) comes
    from rena_bringup/config/config.yaml, keyed by serial.

    Mode is per-camera from config.yaml (image_mode: "raw" | "rect"). cuVSLAM's
    rectified_stereo_camera is a global rig flag, so all OAKs listed in the
    rig must share the same mode — we error otherwise.

    Images are fed to tracker.track() in order:
        [cam0_left, cam0_right, cam1_left, cam1_right, ...]
    """

    def __init__(self, rig_file: str = DEFAULT_OAK_RIG_FILE) -> None:
        self._rig_file = rig_file
        self._stereo_cameras: List[Dict] = []
        self._rect_mode: bool = False
        self._running = False

    @property
    def num_viz_cameras(self) -> int:
        return len(self._stereo_cameras)

    def get_viz_image_indices(self) -> List[int]:
        return list(range(0, len(self._stereo_cameras) * 2, 2))

    def get_viz_observation_indices(self) -> List[int]:
        return list(range(0, len(self._stereo_cameras) * 2, 2))

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        with open(self._rig_file, "r") as f:
            rig_data = yaml.safe_load(f) or {}

        rig_entries = rig_data.get("stereo_cameras", []) or []
        if not rig_entries:
            raise RuntimeError(f"No `stereo_cameras` in {self._rig_file}")

        # Join rig entries with config.yaml (keyed by serial_no).
        stereo_cameras = []
        modes_seen = set()
        for entry in rig_entries:
            serial = entry.get("serial_no")
            if not serial:
                raise RuntimeError(
                    f"oak_rig.yaml entry missing `serial_no`: {entry}"
                )
            cfg_entry = _oak_entry_by_serial(serial)
            modes_seen.add(cfg_entry["image_mode"])
            left_topic, right_topic = _oak_topics_for_entry(cfg_entry)
            stereo_cameras.append({
                "serial_no": serial,
                "key": cfg_entry["key"],
                "robot_part": cfg_entry["robot_part"],
                "image_mode": cfg_entry["image_mode"],
                "rect_calib": cfg_entry["rect_calib"],
                "left_transform": entry["left_camera"]["transform"],
                "right_transform": entry["right_camera"]["transform"],
                "left_topic": left_topic,
                "right_topic": right_topic,
            })

        # cuVSLAM's rectified_stereo_camera is a rig-wide flag; all cams must agree.
        if len(modes_seen) > 1:
            raise RuntimeError(
                f"Mixed image_mode values across OAK rig entries: {sorted(modes_seen)}. "
                "cuVSLAM requires all cameras to share a single rectified_stereo_camera "
                "mode. Set every OAK's image_mode in config.yaml to the same value."
            )
        self._rect_mode = modes_seen.pop() == "rect"
        self._stereo_cameras = stereo_cameras

        print(f"[oak_multicam] mode={'RECT' if self._rect_mode else 'RAW'}")
        for i, c in enumerate(stereo_cameras):
            if self._rect_mode and not c["rect_calib"]:
                raise RuntimeError(
                    f"rect mode: OAK {c['serial_no']} has no rect_calib in "
                    "rena_bringup config.yaml."
                )
            print(f"  cam{i}: serial={c['serial_no']} {c['robot_part']}/{c['key']}  "
                  f"topics: {c['left_topic']} | {c['right_topic']}")

        if self._rect_mode:
            # No camera_info subscription needed in rect mode.
            return {"stereo_cameras": stereo_cameras}

        # Raw mode: subscribe to every camera_info topic.
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        keys = []
        for i in range(len(stereo_cameras)):
            keys += [f"cam{i}_left", f"cam{i}_right"]

        print(f"[oak_multicam] Waiting for raw CameraInfo on {len(keys)} topics ...")
        collector = _CameraInfoCollector(keys)
        node = Node("oak_multicam_camera_info")
        for i, cam in enumerate(stereo_cameras):
            left_info = _oak_image_to_raw_camera_info_topic(cam["left_topic"])
            right_info = _oak_image_to_raw_camera_info_topic(cam["right_topic"])
            node.create_subscription(
                CameraInfo, left_info, partial(collector.on_info, f"cam{i}_left"), 10
            )
            node.create_subscription(
                CameraInfo, right_info, partial(collector.on_info, f"cam{i}_right"), 10
            )

        deadline = time.time() + 5.0
        while not collector.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()

        missing = [k for k, v in collector.received.items() if v is None]
        if missing:
            raise TimeoutError(f"Did not receive CameraInfo for {missing} within 5 s")
        print("[oak_multicam] raw CameraInfo received for all cameras")

        for i, cam in enumerate(stereo_cameras):
            cam["left_msg"] = collector.received[f"cam{i}_left"]
            cam["right_msg"] = collector.received[f"cam{i}_right"]

        return {"stereo_cameras": stereo_cameras}

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        cfg = vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=self._rect_mode,
            use_denoising=False,
        )
        return cfg

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        import numpy as np
        from scipy.spatial.transform import Rotation

        def _mk_rect(calib, transform) -> vslam.Camera:
            c = vslam.Camera()
            c.focal = (calib["fx"], calib["fy"])
            c.principal = (calib["cx"], calib["cy"])
            c.size = (calib["width"], calib["height"])
            c.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
            m = np.asarray(transform, dtype=np.float64)
            c.rig_from_camera = vslam.Pose(
                rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                translation=m[:3, 3],
            )
            return c

        cameras = []
        for cam in camera_params["stereo_cameras"]:
            if self._rect_mode:
                cameras.append(_mk_rect(cam["rect_calib"], cam["left_transform"]))
                cameras.append(_mk_rect(cam["rect_calib"], cam["right_transform"]))
            else:
                cameras.append(_make_oak_raw_camera(
                    k=cam["left_msg"].k, d=cam["left_msg"].d,
                    width=cam["left_msg"].width, height=cam["left_msg"].height,
                    rig_from_camera_4x4=cam["left_transform"],
                ))
                cameras.append(_make_oak_raw_camera(
                    k=cam["right_msg"].k, d=cam["right_msg"].d,
                    width=cam["right_msg"].width, height=cam["right_msg"].height,
                    rig_from_camera_4x4=cam["right_transform"],
                ))
        rig = vslam.Rig()
        rig.cameras = cameras
        return rig

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import message_filters
        from sensor_msgs.msg import CompressedImage
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._running = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        frames_fed = [0]
        last_log_time = [time.monotonic()]

        def on_frames(*msgs):
            ts = msgs[0].header.stamp.sec * 1_000_000_000 + msgs[0].header.stamp.nanosec

            t_decode = time.monotonic()
            images = [_decode_oak_compressed_image(m) for m in msgs]
            decode_ms = (time.monotonic() - t_decode) * 1000

            if any(img is None for img in images):
                print("[oak_multicam] Warning: decode failed", flush=True)
                return

            t0 = time.monotonic()
            vo_pose_estimate, slam_pose = tracker.track(ts, images)
            track_ms = (time.monotonic() - t0) * 1000

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[oak_multicam] Warning: track failed", flush=True)
                return

            frames_fed[0] += 1
            now = time.monotonic()
            if now - last_log_time[0] >= 1.0:
                print(
                    f"[oak_multicam] fed={frames_fed[0]}/s  "
                    f"decode={decode_ms:.1f}ms  track={track_ms:.1f}ms",
                    flush=True,
                )
                frames_fed[0] = 0
                last_log_time[0] = now

            landmarks = [
                Landmark(lm.id, lm.coords) for lm in tracker.get_last_landmarks()
            ]
            output_queue.put(
                TrackingResult(
                    ts,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose),
                    images,
                    landmarks,
                )
            )

        self._node = Node("oak_multicam_frames")
        subscribers = []
        topics_log = []
        for cam in self._stereo_cameras:
            for topic in (cam["left_topic"], cam["right_topic"]):
                compressed = _oak_image_to_compressed_topic(topic)
                subscribers.append(
                    message_filters.Subscriber(
                        self._node, CompressedImage, compressed, qos_profile=qos
                    )
                )
                topics_log.append(compressed)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=10, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_frames)

        print(
            f"[oak_multicam] Subscribed (ApproximateTimeSynchronizer slop={SLOP_SEC*1000:.0f}ms): "
            + ", ".join(topics_log),
            flush=True,
        )

        self._spin_thread = threading.Thread(
            target=_spin_ros_node, args=(self._node, self), daemon=True
        )
        self._spin_thread.start()

    def stop_streaming(self) -> None:
        self._running = False
        if hasattr(self, "_spin_thread"):
            self._spin_thread.join(timeout=2.0)
        if hasattr(self, "_node"):
            self._node.destroy_node()


DEFAULT_OAK_LEFT_TOPIC = "/oak_base_front/left/image_raw"
DEFAULT_OAK_RIGHT_TOPIC = "/oak_base_front/right/image_raw"


def _load_rena_oak_cameras():
    """Scan rena_bringup/config/config.yaml for all OAK cameras across all
    robots, return a flat list of dicts:

        [{"serial_no": str, "key": str, "robot_part": "base"|"arm",
          "image_mode": "raw"|"rect", "rect_calib": dict|None}, ...]

    OAK serial numbers are globally unique across devices, so scanning all
    robots in the file is safe. Pycuvslam runs on a specific robot but doesn't
    need to know which one — the serial alone identifies the camera.
    """
    import os
    import yaml

    try:
        from ament_index_python.packages import get_package_share_directory
        base = get_package_share_directory("rena_bringup")
        config_path = os.path.join(base, "config", "config.yaml")
    except Exception:
        # Dev fallback when running outside a sourced ROS install.
        config_path = "/mnt/jetson_data/rena-control/src/rena_bringup/config/config.yaml"

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    out = []
    for _robot_id, robot_cfg in data.items():
        if not isinstance(robot_cfg, dict):
            continue
        for part in ("base", "arm"):
            part_cfg = robot_cfg.get(part) or {}
            for cam in part_cfg.get("cameras", []) or []:
                if cam.get("type") != "oak":
                    continue
                out.append({
                    "serial_no": cam.get("serial_no"),
                    "key": cam.get("key"),
                    "robot_part": part,
                    "image_mode": cam.get("image_mode", "raw"),
                    "rect_calib": cam.get("rect_calib"),
                    "rotation": cam.get("rotation"),
                })
    return out


def _oak_entry_by_serial(serial_no: str) -> dict:
    """Look up an OAK config entry by serial. Raises if not found."""
    for entry in _load_rena_oak_cameras():
        if entry["serial_no"] == serial_no:
            return entry
    raise RuntimeError(
        f"OAK serial {serial_no!r} not found in rena_bringup config.yaml. "
        f"Add it under the robot's base.cameras list with type=oak and a "
        f"rect_calib block if running in rect mode."
    )


def _oak_topics_for_entry(entry: dict) -> tuple[str, str]:
    """Return (left_topic, right_topic) for a config.yaml OAK entry, using its
    image_mode to pick the image_raw / image_rect suffix."""
    suffix = f"image_{entry['image_mode']}"  # image_raw or image_rect
    ns = f"/oak_{entry['robot_part']}_{entry['key']}"
    return f"{ns}/left/{suffix}", f"{ns}/right/{suffix}"


class RosOakStereoTracker(BaseTracker):
    """Stereo tracking from a single OAK stereo pair via ROS compressed image topics.

    Single source of truth: rena_bringup/config/config.yaml. We look up the
    (single) OAK entry there to get image_mode, key, robot_part, rect_calib,
    and derive the image topics ourselves. The optional constructor topic
    args are retained for testing / overrides.

    image_mode="raw":  subscribe to raw CameraInfo; cuVSLAM rectifies internally
                       using K, D + tf extrinsic; rectified_stereo_camera=False.
    image_mode="rect": use rect_calib from config.yaml for both L/R intrinsics;
                       no CameraInfo subscription; rectified_stereo_camera=True.
    """

    def __init__(
        self,
        left_topic: Optional[str] = None,
        right_topic: Optional[str] = None,
    ) -> None:
        # Resolve the single OAK from config.yaml; topics & mode derive from there.
        oaks = _load_rena_oak_cameras()
        if len(oaks) == 0:
            raise RuntimeError(
                "RosOakStereoTracker: no OAK cameras found in rena_bringup config.yaml."
            )
        if len(oaks) > 1:
            serials = [o["serial_no"] for o in oaks]
            raise RuntimeError(
                "RosOakStereoTracker: multiple OAK cameras found in rena_bringup "
                f"config.yaml (serials: {serials}) — ambiguous for the single-pair "
                "stereo tracker. Use ros_oak_multicam, or reduce config.yaml to a "
                "single OAK entry."
            )
        self._entry = oaks[0]
        self._rect_cam_info = self._entry["image_mode"] == "rect"

        derived_left, derived_right = _oak_topics_for_entry(self._entry)
        self._left_topic = left_topic or derived_left
        self._right_topic = right_topic or derived_right
        self._running = False
        print(f"[ros_oak_stereo] OAK serial={self._entry['serial_no']} "
              f"{self._entry['robot_part']}/{self._entry['key']}  "
              f"mode={self._entry['image_mode']}  "
              f"topics: {self._left_topic} | {self._right_topic}")

    @property
    def num_viz_cameras(self) -> int:
        return 1

    def get_viz_image_indices(self) -> List[int]:
        return [0]

    def get_viz_observation_indices(self) -> List[int]:
        return [0]

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        if self._rect_cam_info:
            calib = self._entry["rect_calib"]
            if calib is None:
                raise RuntimeError(
                    f"OAK serial {self._entry['serial_no']} in config.yaml is "
                    f"missing its rect_calib block (required for rect mode)."
                )
            print(f"[ros_oak_stereo] rect mode: "
                  f"fx={calib['fx']:.2f} cx={calib['cx']:.2f} "
                  f"baseline={calib['baseline_m']:.4f} m")
            return {"calib": calib}

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        left_info_topic = _oak_image_to_raw_camera_info_topic(self._left_topic)
        right_info_topic = _oak_image_to_raw_camera_info_topic(self._right_topic)
        print(
            f"[ros_oak_stereo] Waiting for raw CameraInfo on {left_info_topic}, {right_info_topic} ..."
        )

        collector = _CameraInfoCollector(["left", "right"])
        node = Node("ros_oak_stereo_camera_info")
        node.create_subscription(
            CameraInfo, left_info_topic, partial(collector.on_info, "left"), 10
        )
        node.create_subscription(
            CameraInfo, right_info_topic, partial(collector.on_info, "right"), 10
        )

        deadline = time.time() + 5.0
        while not collector.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()

        missing = [k for k, v in collector.received.items() if v is None]
        if missing:
            raise TimeoutError(f"Did not receive CameraInfo for {missing} within 5 s")
        print("[ros_oak_stereo] raw CameraInfo received")

        left_msg = collector.received["left"]
        right_msg = collector.received["right"]

        # Actual (unrectified) pose of right-cam-frame in left-cam-frame.
        right_in_left_4x4 = _oak_extrinsic_from_tf(
            left_frame=left_msg.header.frame_id,
            right_frame=right_msg.header.frame_id,
        )
        print(
            f"[ros_oak_stereo] extrinsic tf {left_msg.header.frame_id} -> "
            f"{right_msg.header.frame_id}: t={right_in_left_4x4[:3, 3].tolist()}"
        )

        return {
            "left_msg": left_msg,
            "right_msg": right_msg,
            "right_in_left": right_in_left_4x4,
        }

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        cfg = vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=self._rect_cam_info,
            use_denoising=False,
        )
        return cfg

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        import numpy as np
        from scipy.spatial.transform import Rotation

        if self._rect_cam_info:
            # rig_from_left is an optional roll/pitch/yaw (deg) from config.yaml,
            # letting us mount the camera tilted relative to the rig frame.
            rot_cfg = self._entry.get("rotation") or {}
            rpy_deg = [
                float(rot_cfg.get("roll", 0.0)),
                float(rot_cfg.get("pitch", 0.0)),
                float(rot_cfg.get("yaw", 0.0)),
            ]
            rig_from_left = np.eye(4)
            rig_from_left[:3, :3] = Rotation.from_euler(
                "xyz", rpy_deg, degrees=True
            ).as_matrix()
            print(f"[ros_oak_stereo] rect rig_from_left rpy(deg)={rpy_deg}")

            calib = camera_params["calib"]
            # Both cameras share the rectified virtual K; no distortion; right
            # is offset by baseline along +X (no rotation, coplanar after rect).
            def _mk(rig_from_cam_4x4):
                cam = vslam.Camera()
                cam.focal = (calib["fx"], calib["fy"])
                cam.principal = (calib["cx"], calib["cy"])
                cam.size = (calib["width"], calib["height"])
                cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
                m = np.asarray(rig_from_cam_4x4, dtype=np.float64)
                cam.rig_from_camera = vslam.Pose(
                    rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                    translation=m[:3, 3],
                )
                return cam

            right_pose = np.eye(4)
            right_pose[0, 3] = calib["baseline_m"]
            rig = vslam.Rig()
            rig.cameras = [_mk(rig_from_left), _mk(rig_from_left @ right_pose)]
            return rig

        # Raw mode: left is the rig origin; rotation is not supported here.
        left_msg = camera_params["left_msg"]
        right_msg = camera_params["right_msg"]
        right_in_left = camera_params["right_in_left"]

        left_cam = _make_oak_raw_camera(
            k=left_msg.k, d=left_msg.d,
            width=left_msg.width, height=left_msg.height,
            rig_from_camera_4x4=np.eye(4),
        )
        right_cam = _make_oak_raw_camera(
            k=right_msg.k, d=right_msg.d,
            width=right_msg.width, height=right_msg.height,
            rig_from_camera_4x4=right_in_left,
        )
        rig = vslam.Rig()
        rig.cameras = [left_cam, right_cam]
        return rig

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import message_filters
        from sensor_msgs.msg import CompressedImage
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._running = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        frames_fed = [0]
        last_log_time = [time.monotonic()]
        last_ts = [0]

        def on_stereo_pair(left_msg, right_msg):
            ts = (
                left_msg.header.stamp.sec * 1_000_000_000
                + left_msg.header.stamp.nanosec
            )

            if ts <= last_ts[0]:
                return
            last_ts[0] = ts

            images = [
                _decode_oak_compressed_image(left_msg),
                _decode_oak_compressed_image(right_msg),
            ]
            if any(img is None for img in images):
                print("[ros_oak_stereo] Warning: decode failed", flush=True)
                return

            t0 = time.monotonic()
            vo_pose_estimate, slam_pose = tracker.track(ts, images)
            track_ms = (time.monotonic() - t0) * 1000

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[ros_oak_stereo] Warning: track failed", flush=True)
                return

            frames_fed[0] += 1
            now = time.monotonic()
            if now - last_log_time[0] >= 1.0:
                print(
                    f"[ros_oak_stereo] fed={frames_fed[0]}/s  track={track_ms:.1f}ms",
                    flush=True,
                )
                frames_fed[0] = 0
                last_log_time[0] = now

            landmarks = [
                Landmark(lm.id, lm.coords) for lm in tracker.get_last_landmarks()
            ]
            output_queue.put(
                TrackingResult(
                    ts,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose),
                    images,
                    landmarks,
                )
            )

        left_compressed = _oak_image_to_compressed_topic(self._left_topic)
        right_compressed = _oak_image_to_compressed_topic(self._right_topic)

        self._node = Node("ros_oak_stereo_frames")
        left_sub = message_filters.Subscriber(
            self._node, CompressedImage, left_compressed, qos_profile=qos
        )
        right_sub = message_filters.Subscriber(
            self._node, CompressedImage, right_compressed, qos_profile=qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=10, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_stereo_pair)

        print(
            f"[ros_oak_stereo] Subscribed (ApproximateTimeSynchronizer slop={SLOP_SEC*1000:.0f}ms): "
            f"{left_compressed}, {right_compressed}",
            flush=True,
        )

        self._spin_thread = threading.Thread(
            target=_spin_ros_node, args=(self._node, self), daemon=True
        )
        self._spin_thread.start()

    def stop_streaming(self) -> None:
        self._running = False
        if hasattr(self, "_spin_thread"):
            self._spin_thread.join(timeout=2.0)
        if hasattr(self, "_node"):
            self._node.destroy_node()
