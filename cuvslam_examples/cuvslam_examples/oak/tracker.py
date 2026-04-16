import queue
import threading
import time
from functools import partial
from typing import Dict, List

import yaml

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.camera_utils import get_rs_multi_rig
from cuvslam_examples.realsense.tracker import (
    BaseTracker,
    _CameraInfoCollector,
    _make_intrinsics_like,
    _spin_ros_node,
)
from cuvslam_examples.realsense.utils import Landmark, Pose

DEFAULT_OAK_RIG_FILE = "oak_rig.yaml"

SLOP_SEC = 0.002  # 2 ms


def _oak_image_to_camera_info_topic(image_topic: str) -> str:
    """OAK convention: /oak_base_front/left/image_rect -> /oak_base_front/left/camera_info"""
    parent = image_topic.rsplit("/", 1)[0]
    return parent + "/camera_info"


def _oak_image_to_compressed_topic(image_topic: str) -> str:
    """OAK images are always recorded as compressed: /oak_base_front/left/image_rect/compressed"""
    return image_topic + "/compressed"


def _decode_oak_compressed_image(msg) -> "np.ndarray | None":
    import cv2
    import numpy as np

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return np.ascontiguousarray(img) if img is not None else None


class RosOakMulticamTracker(BaseTracker):
    """Multi-camera stereo tracking from N OAK-D stereo pairs via ROS topics.

    Intrinsics are read from camera_info topics at startup.
    Extrinsics (rig_from_camera transforms) are loaded from oak_rig.yaml.

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

        print(f"[oak_multicam] Waiting for CameraInfo on {len(keys)} topics ...")

        collector = _CameraInfoCollector(keys)
        node = Node("oak_multicam_camera_info")

        for i, cam in enumerate(self._stereo_cameras):
            left_info = _oak_image_to_camera_info_topic(cam["left_topic"])
            right_info = _oak_image_to_camera_info_topic(cam["right_topic"])
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
        print("[oak_multicam] CameraInfo received for all cameras")

        camera_params: Dict[str, Dict] = {}
        for i, cam in enumerate(self._stereo_cameras):
            left_msg = collector.received[f"cam{i}_left"]
            right_msg = collector.received[f"cam{i}_right"]
            camera_params[f"camera_{i + 1}"] = {
                "left": {
                    "intrinsics": _make_intrinsics_like(
                        left_msg.k[0],
                        left_msg.k[4],
                        left_msg.k[2],
                        left_msg.k[5],
                        left_msg.width,
                        left_msg.height,
                    ),
                    "extrinsics": cam["left_camera"]["transform"],
                },
                "right": {
                    "intrinsics": _make_intrinsics_like(
                        right_msg.k[0],
                        right_msg.k[4],
                        right_msg.k[2],
                        right_msg.k[5],
                        right_msg.width,
                        right_msg.height,
                    ),
                    "extrinsics": cam["right_camera"]["transform"],
                },
            }

        return camera_params

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=True,
            use_denoising=False,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_multi_rig(camera_params)

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
