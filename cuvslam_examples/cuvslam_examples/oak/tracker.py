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

SLOP_SEC = 0.002  # 2 ms


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

    Consumes RAW (distorted) image_raw streams plus raw CameraInfo (real K, D).
    cuVSLAM is configured with rectified_stereo_camera=False so it rectifies
    internally using per-camera K, D and each camera's rig_from_camera pose
    taken from oak_rig.yaml. No /tf lookup (unlike stereo tracker) because the
    rig already defines all camera poses explicitly.

    Images are fed to tracker.track() in order:
        [cam0_left, cam0_right, cam1_left, cam1_right, ...]
    """

    def __init__(self, rig_file: str = DEFAULT_OAK_RIG_FILE) -> None:
        self._rig_file = rig_file
        self._stereo_cameras: List[Dict] = []
        self._running = False

    @property
    def num_viz_cameras(self) -> int:
        return len(self._stereo_cameras)

    def get_viz_image_indices(self) -> List[int]:
        return list(range(0, len(self._stereo_cameras) * 2, 2))

    def get_viz_observation_indices(self) -> List[int]:
        return list(range(0, len(self._stereo_cameras) * 2, 2))

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        with open(self._rig_file, "r") as f:
            config_data = yaml.safe_load(f)

        self._stereo_cameras = config_data["stereo_cameras"]

        keys = []
        for i in range(len(self._stereo_cameras)):
            keys += [f"cam{i}_left", f"cam{i}_right"]

        print(f"[oak_multicam] Waiting for raw CameraInfo on {len(keys)} topics ...")

        collector = _CameraInfoCollector(keys)
        node = Node("oak_multicam_camera_info")

        for i, cam in enumerate(self._stereo_cameras):
            left_info = _oak_image_to_raw_camera_info_topic(cam["left_topic"])
            right_info = _oak_image_to_raw_camera_info_topic(cam["right_topic"])
            node.create_subscription(
                CameraInfo, left_info, partial(collector.on_info, f"cam{i}_left"), 10
            )
            node.create_subscription(
                CameraInfo, right_info, partial(collector.on_info, f"cam{i}_right"), 10
            )

        deadline = time.time() + 30.0
        while not collector.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()

        missing = [k for k, v in collector.received.items() if v is None]
        if missing:
            raise TimeoutError(f"Did not receive CameraInfo for {missing} within 30 s")
        print("[oak_multicam] raw CameraInfo received for all cameras")

        stereo_pairs = []
        for i, cam in enumerate(self._stereo_cameras):
            left_msg = collector.received[f"cam{i}_left"]
            right_msg = collector.received[f"cam{i}_right"]

            stereo_pairs.append({
                "left_msg": left_msg,
                "right_msg": right_msg,
                "left_transform": cam["left_camera"]["transform"],
                "right_transform": cam["right_camera"]["transform"],
            })

        return {"stereo_pairs": stereo_pairs}

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        cfg = vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=False,
            use_denoising=False,
        )
        return cfg

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        cameras = []
        for pair in camera_params["stereo_pairs"]:
            cameras.append(_make_oak_raw_camera(
                k=pair["left_msg"].k, d=pair["left_msg"].d,
                width=pair["left_msg"].width, height=pair["left_msg"].height,
                rig_from_camera_4x4=pair["left_transform"],
            ))
            cameras.append(_make_oak_raw_camera(
                k=pair["right_msg"].k, d=pair["right_msg"].d,
                width=pair["right_msg"].width, height=pair["right_msg"].height,
                rig_from_camera_4x4=pair["right_transform"],
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


class RosOakStereoTracker(BaseTracker):
    """Stereo tracking from a single OAK stereo pair via ROS compressed image topics.

    Consumes RAW (distorted, unrectified) left/right image streams plus the raw
    CameraInfo (real K, D). cuVSLAM is configured with rectified_stereo_camera=False
    so it rectifies internally using K, D and the actual inter-camera extrinsic.
    Extrinsic is read from /tf (frame_ids of the two CameraInfo messages).
    """

    def __init__(
        self,
        left_topic: str = DEFAULT_OAK_LEFT_TOPIC,
        right_topic: str = DEFAULT_OAK_RIGHT_TOPIC,
    ) -> None:
        self._left_topic = left_topic
        self._right_topic = right_topic
        self._running = False

    @property
    def num_viz_cameras(self) -> int:
        return 1

    def get_viz_image_indices(self) -> List[int]:
        return [0]

    def get_viz_observation_indices(self) -> List[int]:
        return [0]

    def setup_camera_parameters(self) -> Dict[str, Dict]:
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

        deadline = time.time() + 30.0
        while not collector.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()

        missing = [k for k, v in collector.received.items() if v is None]
        if missing:
            raise TimeoutError(f"Did not receive CameraInfo for {missing} within 30 s")
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
            rectified_stereo_camera=False,
            use_denoising=False,
        )
        return cfg

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        import numpy as np

        left_msg = camera_params["left_msg"]
        right_msg = camera_params["right_msg"]
        right_in_left = camera_params["right_in_left"]

        left_cam = _make_oak_raw_camera(
            k=left_msg.k, d=left_msg.d,
            width=left_msg.width, height=left_msg.height,
            rig_from_camera_4x4=np.eye(4),  # left is the rig origin
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
