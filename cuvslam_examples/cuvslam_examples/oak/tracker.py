import queue
import threading
import time
from functools import partial
from typing import Dict, List

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

SLOP_SEC = 0.001


# Rotation that maps robot body axes (x-fwd, y-left, z-up) into cuVSLAM
# optical axes (x-right, y-down, z-fwd). Rows are optical basis vectors
# expressed in robot coordinates; equivalently,
#   R_OPT_FROM_ROBOT @ [x_fwd, y_left, z_up] = [-y, -z, x] = [right, down, fwd].
_R_OPT_FROM_ROBOT = [
    [0, -1, 0],
    [0,  0, -1],
    [1,  0, 0],
]


def _rig_from_camera_from_robot_pose(rotation_cfg: dict, translation_cfg):
    """Convert a camera pose given in robot body frame to the equivalent
    rig_from_camera 4x4 in cuVSLAM's optical rig convention.

    The rig frame is kept in optical axes (x-right, y-down, z-fwd) so that the
    default ``Pose.to_robot_frame()`` (cuVSLAM -> ROS) still applies to the
    tracker output. Config values are intuitive robot-frame angles/offsets,
    which we rebase here:

        R_rig = R_OPT_FROM_ROBOT @ R_robot @ R_OPT_FROM_ROBOT.T
        t_rig = R_OPT_FROM_ROBOT @ t_robot
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    rpy = [
        float((rotation_cfg or {}).get("roll", 0.0)),
        float((rotation_cfg or {}).get("pitch", 0.0)),
        float((rotation_cfg or {}).get("yaw", 0.0)),
    ]
    t_robot = np.asarray(translation_cfg or [0.0, 0.0, 0.0], dtype=np.float64)
    if t_robot.shape != (3,):
        raise ValueError(
            f"rig.translation must be a 3-element list, got {translation_cfg!r}"
        )

    R_robot = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    C = np.asarray(_R_OPT_FROM_ROBOT, dtype=np.float64)
    R_rig = C @ R_robot @ C.T
    t_rig = C @ t_robot

    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R_rig
    m[:3, 3] = t_rig
    return m


def _oak_image_to_compressed_topic(image_topic: str) -> str:
    """OAK images are recorded as compressed: /oak_base_front/left/image_raw/compressed"""
    return image_topic + "/compressed"


def _decode_oak_compressed_image(msg) -> "np.ndarray | None":
    import cv2
    import numpy as np

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return np.ascontiguousarray(img) if img is not None else None


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
                    "rig": cam.get("rig"),
                })
    return out


def _oak_topics_for_entry(entry: dict) -> tuple[str, str]:
    """Return (left_topic, right_topic) for a config.yaml OAK entry, using its
    image_mode to pick the image_raw / image_rect suffix."""
    suffix = f"image_{entry['image_mode']}"  # image_raw or image_rect
    ns = f"/oak_{entry['robot_part']}_{entry['key']}"
    return f"{ns}/left/{suffix}", f"{ns}/right/{suffix}"


class RosOakStereoTracker(BaseTracker):
    """Unified OAK stereo tracker — single or multi-camera.

    Reads every OAK entry from rena_bringup/config/config.yaml and auto-picks
    cuVSLAM stereo (1 pair) vs multicam (N pairs). All inter-unit extrinsics
    live in config.yaml under each entry's ``rig:`` block (translation +
    rotation in robot body frame, x-fwd/y-left/z-up); the tracker rebases that
    into cuVSLAM's optical rig convention before calling ``track()``. The
    within-unit L/R baseline always comes from ``rect_calib.baseline_m``.

    Images are fed to tracker.track() in order:
        [cam0_left, cam0_right, cam1_left, cam1_right, ...]
    """

    def __init__(self) -> None:
        oaks = _load_rena_oak_cameras()
        if not oaks:
            raise RuntimeError(
                "RosOakStereoTracker: no OAK cameras found in rena_bringup config.yaml."
            )

        # cuVSLAM's rectified_stereo_camera is a rig-wide flag; all OAKs must
        # agree on image_mode.
        modes = {o["image_mode"] for o in oaks}
        if len(modes) > 1:
            raise RuntimeError(
                f"RosOakStereoTracker: mixed image_mode across OAKs ({sorted(modes)}). "
                "cuVSLAM requires all cameras to share one rectified_stereo_camera "
                "setting — make every OAK's image_mode match."
            )
        self._rect_cam_info = modes.pop() == "rect"
        self._entries = oaks
        for e in self._entries:
            left_topic, right_topic = _oak_topics_for_entry(e)
            e["left_topic"] = left_topic
            e["right_topic"] = right_topic

        mode_label = "RECT" if self._rect_cam_info else "RAW"
        print(f"[ros_oak_stereo] mode={mode_label}  N={len(self._entries)}")
        for i, e in enumerate(self._entries):
            print(
                f"  cam{i}: serial={e['serial_no']} {e['robot_part']}/{e['key']}  "
                f"topics: {e['left_topic']} | {e['right_topic']}"
            )

        self._running = False

    @property
    def num_viz_cameras(self) -> int:
        return len(self._entries)

    def get_viz_image_indices(self) -> List[int]:
        return list(range(0, len(self._entries) * 2, 2))

    def get_viz_observation_indices(self) -> List[int]:
        return list(range(0, len(self._entries) * 2, 2))

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        per_entry: List[Dict] = []

        if self._rect_cam_info:
            for entry in self._entries:
                calib = entry["rect_calib"]
                if calib is None:
                    raise RuntimeError(
                        f"OAK serial {entry['serial_no']} in config.yaml is "
                        f"missing its rect_calib block (required for rect mode)."
                    )
                per_entry.append({"entry": entry, "calib": calib})
            return {"entries": per_entry}

        # Raw mode: batch-collect every CameraInfo pair. Baseline comes from
        # rect_calib.baseline_m (not /tf) — the TF baseline can drift; rect_calib
        # is the factory stereo calibration and keeps the scale correct.
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        keys: List[str] = []
        for i in range(len(self._entries)):
            keys += [f"cam{i}_left", f"cam{i}_right"]

        print(
            f"[ros_oak_stereo] Waiting for raw CameraInfo on {len(keys)} topics ..."
        )
        collector = _CameraInfoCollector(keys)
        node = Node("ros_oak_stereo_camera_info")
        for i, entry in enumerate(self._entries):
            left_info = _oak_image_to_raw_camera_info_topic(entry["left_topic"])
            right_info = _oak_image_to_raw_camera_info_topic(entry["right_topic"])
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
            raise TimeoutError(
                f"Did not receive CameraInfo for {missing} within 5 s"
            )
        print("[ros_oak_stereo] raw CameraInfo received for all cameras")

        for i, entry in enumerate(self._entries):
            if entry["rect_calib"] is None:
                raise RuntimeError(
                    f"OAK serial {entry['serial_no']} in config.yaml is missing "
                    "its rect_calib block (needed for baseline_m in raw mode too)."
                )
            per_entry.append({
                "entry": entry,
                "left_msg": collector.received[f"cam{i}_left"],
                "right_msg": collector.received[f"cam{i}_right"],
            })

        return {"entries": per_entry}

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=self._rect_cam_info,
            use_denoising=False,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=False)

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        import numpy as np
        from scipy.spatial.transform import Rotation

        def _mk_rect(calib: dict, rig_from_cam: "np.ndarray") -> vslam.Camera:
            cam = vslam.Camera()
            cam.focal = (calib["fx"], calib["fy"])
            cam.principal = (calib["cx"], calib["cy"])
            cam.size = (calib["width"], calib["height"])
            cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
            m = np.asarray(rig_from_cam, dtype=np.float64)
            cam.rig_from_camera = vslam.Pose(
                rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                translation=m[:3, 3],
            )
            return cam

        cameras: List[vslam.Camera] = []
        for p in camera_params["entries"]:
            entry = p["entry"]
            rig_cfg = entry.get("rig") or {}
            rig_from_left = _rig_from_camera_from_robot_pose(
                rig_cfg.get("rotation"),
                rig_cfg.get("translation"),
            )
            print(
                f"[ros_oak_stereo] {entry['robot_part']}/{entry['key']}: "
                f"rig_from_left t={rig_from_left[:3, 3].tolist()}"
            )

            # Baseline always comes from rect_calib — in both raw and rect mode
            # we trust the factory stereo calibration over /tf.
            baseline_m = entry["rect_calib"]["baseline_m"]
            baseline = np.eye(4)
            baseline[0, 3] = baseline_m
            rig_from_right = rig_from_left @ baseline

            if self._rect_cam_info:
                calib = p["calib"]
                cameras.append(_mk_rect(calib, rig_from_left))
                cameras.append(_mk_rect(calib, rig_from_right))
            else:
                left_msg = p["left_msg"]
                right_msg = p["right_msg"]
                cameras.append(_make_oak_raw_camera(
                    k=left_msg.k, d=left_msg.d,
                    width=left_msg.width, height=left_msg.height,
                    rig_from_camera_4x4=rig_from_left,
                ))
                cameras.append(_make_oak_raw_camera(
                    k=right_msg.k, d=right_msg.d,
                    width=right_msg.width, height=right_msg.height,
                    rig_from_camera_4x4=rig_from_right,
                ))

        rig = vslam.Rig()
        rig.cameras = cameras
        return rig

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

        def on_frames(*msgs):
            ts = (
                msgs[0].header.stamp.sec * 1_000_000_000
                + msgs[0].header.stamp.nanosec
            )
            if ts <= last_ts[0]:
                return
            last_ts[0] = ts

            t_decode = time.monotonic()
            images = [_decode_oak_compressed_image(m) for m in msgs]
            decode_ms = (time.monotonic() - t_decode) * 1000

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
                    f"[ros_oak_stereo] fed={frames_fed[0]}/s  "
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

        self._node = Node("ros_oak_stereo_frames")
        subscribers = []
        topics_log: List[str] = []
        for entry in self._entries:
            for topic in (entry["left_topic"], entry["right_topic"]):
                compressed = _oak_image_to_compressed_topic(topic)
                subscribers.append(
                    message_filters.Subscriber(
                        self._node, CompressedImage, compressed, qos_profile=qos
                    )
                )
                topics_log.append(compressed)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=100, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_frames)

        print(
            f"[ros_oak_stereo] Subscribed (ApproximateTimeSynchronizer "
            f"slop={SLOP_SEC*1000:.0f}ms): " + ", ".join(topics_log),
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
