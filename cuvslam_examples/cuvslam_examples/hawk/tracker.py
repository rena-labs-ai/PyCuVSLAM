#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import queue
import threading
import time
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import yaml

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.camera_utils import get_rs_multi_rig, get_rs_stereo_rig
from cuvslam_examples.realsense.tracker import (
    BaseTracker,
    _CameraInfoCollector,
    _make_intrinsics_like,
    _spin_ros_node,
)
from cuvslam_examples.realsense.utils import Landmark, Pose

DEFAULT_HAWK_RIG_FILE = "hawk_rig.yaml"  # resolved via hawk_rig_file launch arg; no default path bundled here

DEFAULT_HAWK_LEFT_TOPIC = "/left/image_rect"
DEFAULT_HAWK_RIGHT_TOPIC = "/right/image_rect"

SLOP_SEC = 0.002  # 2 ms


def _hawk_image_to_camera_info_topic(image_topic: str) -> str:
    """Hawk convention: /left/image_rect -> /left/camera_info_rect"""
    return image_topic.replace("image_rect", "camera_info_rect")


def _decode_hawk_image(msg) -> Optional[np.ndarray]:
    """Decode a raw sensor_msgs/Image to a contiguous numpy array."""
    enc = msg.encoding.lower()
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = raw[: msg.step * msg.height].reshape(msg.height, msg.step)[:, : msg.width * (msg.step // msg.width)]
    if enc == "bgr8":
        img = img.reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(img[:, :, ::-1])  # BGR -> RGB
    if enc == "rgb8":
        return np.ascontiguousarray(img.reshape(msg.height, msg.width, 3))
    # mono8 / grayscale
    return np.ascontiguousarray(img.reshape(msg.height, msg.width))


class RosHawkStereoTracker(BaseTracker):
    """Stereo tracking from Hawk camera ROS CompressedImage topics (rectified)."""

    def __init__(
        self,
        left_topic: str = DEFAULT_HAWK_LEFT_TOPIC,
        right_topic: str = DEFAULT_HAWK_RIGHT_TOPIC,
    ) -> None:
        self._left_topic = left_topic
        self._right_topic = right_topic
        self._running = False

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo
        from functools import partial

        left_info_topic = _hawk_image_to_camera_info_topic(self._left_topic)
        right_info_topic = _hawk_image_to_camera_info_topic(self._right_topic)

        print(
            f"[ros_hawk_stereo] Waiting for CameraInfo on "
            f"{left_info_topic}, {right_info_topic} ..."
        )

        collector = _CameraInfoCollector(["left", "right"])
        node = Node("ros_hawk_stereo_camera_info")
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
        print("[ros_hawk_stereo] CameraInfo received")

        left_msg = collector.received["left"]
        right_msg = collector.received["right"]
        assert left_msg is not None and right_msg is not None

        # Baseline from right camera P matrix: P[0,3] = -fx * baseline
        fx_right = right_msg.p[0]
        baseline = -right_msg.p[3] / fx_right if fx_right != 0 else 0.12

        right_extrinsics = [
            [1, 0, 0, baseline],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]

        return {
            "left": {
                "intrinsics": _make_intrinsics_like(
                    left_msg.k[0],
                    left_msg.k[4],
                    left_msg.k[2],
                    left_msg.k[5],
                    left_msg.width,
                    left_msg.height,
                ),
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
                "extrinsics": right_extrinsics,
            },
        }

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=False,
            use_denoising=False,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_stereo_rig(
            camera_params,
            border_bottom=30,
            border_left=150,
            border_right=30,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import message_filters
        from sensor_msgs.msg import Image
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

        def on_stereo_pair(left_msg, right_msg):
            ts = left_msg.header.stamp.sec * 1_000_000_000 + left_msg.header.stamp.nanosec

            t_decode = time.monotonic()
            images = [_decode_hawk_image(left_msg), _decode_hawk_image(right_msg)]
            decode_ms = (time.monotonic() - t_decode) * 1000

            if any(img is None for img in images):
                print("[ros_hawk_stereo] Warning: decode failed", flush=True)
                return

            t0 = time.monotonic()
            vo_pose_estimate, slam_pose = tracker.track(ts, images)
            track_ms = (time.monotonic() - t0) * 1000

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[ros_hawk_stereo] Warning: track failed", flush=True)
                return

            frames_fed[0] += 1
            now = time.monotonic()
            if now - last_log_time[0] >= 1.0:
                print(
                    f"[ros_hawk_stereo] fed={frames_fed[0]}/s  decode={decode_ms:.1f}ms track={track_ms:.1f}ms",
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

        self._node = Node("ros_hawk_stereo_frames")
        left_sub = message_filters.Subscriber(
            self._node, Image, self._left_topic, qos_profile=qos
        )
        right_sub = message_filters.Subscriber(
            self._node, Image, self._right_topic, qos_profile=qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=10, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_stereo_pair)

        print(
            f"[ros_hawk_stereo] Subscribed (ApproximateTimeSynchronizer slop={SLOP_SEC*1000:.0f}ms): "
            f"{self._left_topic}, {self._right_topic}",
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


class RosHawkMulticamTracker(BaseTracker):
    """Multi-camera stereo tracking from N Hawk stereo pairs via ROS image_rect topics.

    Intrinsics are collected from camera_info_rect topics at startup;
    extrinsics (rig_from_camera transforms) are loaded from a YAML config
    with the same format as frame_agx_rig.yaml.

    Images are fed to tracker.track() in order:
        [cam0_left, cam0_right, cam1_left, cam1_right, ...]
    """

    def __init__(self, rig_file: str = DEFAULT_HAWK_RIG_FILE) -> None:
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

        # Build flat key list: cam0_left, cam0_right, cam1_left, ...
        keys = []
        for i, cam in enumerate(self._stereo_cameras):
            keys += [f"cam{i}_left", f"cam{i}_right"]

        print(f"[hawk_multicam] Waiting for CameraInfo on {len(keys)} topics ...")

        collector = _CameraInfoCollector(keys)
        node = Node("hawk_multicam_camera_info")

        for i, cam in enumerate(self._stereo_cameras):
            left_info = _hawk_image_to_camera_info_topic(cam["left_topic"])
            right_info = _hawk_image_to_camera_info_topic(cam["right_topic"])
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
        print("[hawk_multicam] CameraInfo received for all cameras")

        camera_params: Dict[str, Dict] = {}
        for i, cam in enumerate(self._stereo_cameras):
            left_msg = collector.received[f"cam{i}_left"]
            right_msg = collector.received[f"cam{i}_right"]
            camera_params[f"camera_{i + 1}"] = {
                "left": {
                    "intrinsics": _make_intrinsics_like(
                        left_msg.k[0], left_msg.k[4],
                        left_msg.k[2], left_msg.k[5],
                        left_msg.width, left_msg.height,
                    ),
                    "extrinsics": cam["left_camera"]["transform"],
                },
                "right": {
                    "intrinsics": _make_intrinsics_like(
                        right_msg.k[0], right_msg.k[4],
                        right_msg.k[2], right_msg.k[5],
                        right_msg.width, right_msg.height,
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
        return get_rs_multi_rig(
            camera_params,
            border_bottom=30,
            border_left=150,
            border_right=30,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import message_filters
        from sensor_msgs.msg import Image
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
        num_cams = len(self._stereo_cameras)

        def on_frames(*msgs):
            # msgs order: cam0_left, cam0_right, cam1_left, cam1_right, ...
            ts = msgs[0].header.stamp.sec * 1_000_000_000 + msgs[0].header.stamp.nanosec

            t_decode = time.monotonic()
            images = [_decode_hawk_image(m) for m in msgs]
            decode_ms = (time.monotonic() - t_decode) * 1000

            if any(img is None for img in images):
                print("[hawk_multicam] Warning: decode failed", flush=True)
                return

            t0 = time.monotonic()
            vo_pose_estimate, slam_pose = tracker.track(ts, images)
            track_ms = (time.monotonic() - t0) * 1000

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[hawk_multicam] Warning: track failed", flush=True)
                return

            frames_fed[0] += 1
            now = time.monotonic()
            if now - last_log_time[0] >= 1.0:
                print(
                    f"[hawk_multicam] fed={frames_fed[0]}/s  "
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

        self._node = Node("hawk_multicam_frames")
        subscribers = []
        topics_log = []
        for cam in self._stereo_cameras:
            for topic in (cam["left_topic"], cam["right_topic"]):
                subscribers.append(
                    message_filters.Subscriber(self._node, Image, topic, qos_profile=qos)
                )
                topics_log.append(topic)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=10, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_frames)

        print(
            f"[hawk_multicam] Subscribed (ApproximateTimeSynchronizer slop={SLOP_SEC*1000:.0f}ms): "
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
