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
from typing import Dict, List, Optional

import numpy as np
import pyzed.sl as sl

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.camera_utils import get_rs_stereo_rig
from cuvslam_examples.realsense.tracker import (
    BaseTracker,
    _CameraInfoCollector,
    _FrameAggregator,
    _make_intrinsics_like,
    _spin_ros_node,
)
from cuvslam_examples.realsense.utils import Landmark, Pose

from cuvslam_examples.zed.camera_utils import get_zed_stereo_rig, setup_zed_camera

RESOLUTION = (640, 480)
FPS = 60
WARMUP_FRAMES = 10
IMAGE_JITTER_THRESHOLD_NS = ((1000 / FPS) + 2) * 1e6

# ZED ROS2: camera_info is at {image_topic}/camera_info (e.g. .../color/rect/image/camera_info)
DEFAULT_ZED_LEFT_TOPIC = "/zed/zed_node/left/color/rect/image/compressed"
DEFAULT_ZED_RIGHT_TOPIC = "/zed/zed_node/right/color/rect/image/compressed"


def _zed_image_to_camera_info_topic(image_topic: str) -> str:
    """ZED convention: .../image/compressed -> .../image/camera_info"""
    base = image_topic.replace("/compressed", "").rstrip("/")
    return f"{base}/camera_info"


def _decode_zed_compressed_image(raw_bytes: bytes) -> Optional[np.ndarray]:
    """Decode ZED compressed image to RGB, contiguous (matches ZedStereoTracker format)."""
    import cv2

    img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # BGR -> RGB, ensure contiguous (cuvslam requires stride(1)==1)
    return np.ascontiguousarray(img[:, :, [2, 1, 0]])


class ZedStereoTracker(BaseTracker):
    """Stereo tracking strategy for ZED camera."""

    def __init__(self, raw: bool = False) -> None:
        self._zed: Optional[sl.Camera] = None
        self._running = False
        self._raw = raw

    def setup_camera_parameters(self) -> dict:
        zed, camera_info = setup_zed_camera(RESOLUTION, FPS)
        zed.close()
        return {"camera_info": camera_info}

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        cfg = vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
        )
        cfg.horizontal_stereo_camera = not self._raw
        return cfg

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_zed_stereo_rig(camera_params["camera_info"], raw=self._raw)

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        self._zed, _ = setup_zed_camera(RESOLUTION, FPS)
        self._running = True

        threading.Thread(
            target=self._spin,
            args=(tracker, output_queue),
            daemon=True,
        ).start()

    def stop_streaming(self) -> None:
        self._running = False
        if self._zed:
            self._zed.close()
            self._zed = None

    def _spin(self, tracker: vslam.Tracker, output_queue: queue.Queue) -> None:
        frame_id = 0
        prev_timestamp: Optional[int] = None
        dropped_count = 0
        runtime_params = sl.RuntimeParameters()
        image_left = sl.Mat()
        image_right = sl.Mat()

        left_view = sl.VIEW.LEFT_UNRECTIFIED if self._raw else sl.VIEW.LEFT
        right_view = sl.VIEW.RIGHT_UNRECTIFIED if self._raw else sl.VIEW.RIGHT

        while self._running and self._zed:
            if self._zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            timestamp = int(
                self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            )
            frame_id += 1

            if prev_timestamp is not None:
                ts_diff = timestamp - prev_timestamp
                if ts_diff > IMAGE_JITTER_THRESHOLD_NS:
                    dropped_count += 1
                    print(
                        f"Warning: Camera stream message drop: frame_id={frame_id} "
                        f"timestamp gap ({ts_diff/1e6:.2f} ms) exceeds "
                        f"threshold {IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms "
                        f"(total_dropped={dropped_count})",
                        flush=True,
                    )
            prev_timestamp = timestamp

            self._zed.retrieve_image(image_left, left_view)
            self._zed.retrieve_image(image_right, right_view)
            left_data = image_left.get_data()
            right_data = image_right.get_data()
            images = (
                np.ascontiguousarray(left_data[:, :, [2, 1, 0]]),
                np.ascontiguousarray(right_data[:, :, [2, 1, 0]]),
            )

            if frame_id <= WARMUP_FRAMES:
                continue

            vo_pose_estimate, slam_pose = tracker.track(timestamp, images)

            if vo_pose_estimate.world_from_rig is None:
                print("Warning: VO pose tracking not valid")
                continue

            if slam_pose is None:
                print("Warning: SLAM pose tracking not valid")
                continue

            landmarks = [
                Landmark(lm.id, lm.coords) for lm in tracker.get_last_landmarks()
            ]

            output_queue.put(
                TrackingResult(
                    timestamp,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose),
                    images,
                    landmarks,
                )
            )


class RosZedStereoTracker(BaseTracker):
    """Stereo tracking from ZED ROS CompressedImage topics."""

    def __init__(
        self,
        left_topic: str,
        right_topic: str,
        use_compressed: bool = True,
    ) -> None:
        if not use_compressed:
            raise ValueError("Non-compressed (raw Image) mode not supported")
        self._left_topic = left_topic
        self._right_topic = right_topic
        self._running = False

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        import time
        from functools import partial

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        left_info_topic = _zed_image_to_camera_info_topic(self._left_topic)
        right_info_topic = _zed_image_to_camera_info_topic(self._right_topic)

        print(
            f"[ros_zed_stereo] Waiting for CameraInfo on {left_info_topic}, {right_info_topic} ..."
        )

        collector = _CameraInfoCollector(["left", "right"])
        node = Node("ros_zed_stereo_camera_info")
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
        print("[ros_zed_stereo] CameraInfo received")

        left_msg = collector.received["left"]
        right_msg = collector.received["right"]
        assert left_msg is not None and right_msg is not None

        # Baseline from right camera P matrix: P[0,3] = -fx*baseline
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
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        h = camera_params["left"]["intrinsics"].height
        return get_rs_stereo_rig(
            camera_params,
            border_bottom=h // 4,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        from sensor_msgs.msg import CompressedImage

        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._running = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        aggregator = _FrameAggregator(
            2, tracker, output_queue, decode_fn=_decode_zed_compressed_image
        )
        first_received = {"left": True, "right": True}
        counts = {"left": 0, "right": 0}

        def on_left(msg):
            counts["left"] += 1
            if first_received["left"]:
                print("[ros_zed_stereo] First left frame received", flush=True)
                first_received["left"] = False
            aggregator.on_compressed_msg(0, msg)

        def on_right(msg):
            counts["right"] += 1
            if first_received["right"]:
                print("[ros_zed_stereo] First right frame received", flush=True)
                first_received["right"] = False
            aggregator.on_compressed_msg(1, msg)

        def _log_counts():
            while self._running:
                time.sleep(5.0)
                if self._running and (counts["left"] > 0 or counts["right"] > 0):
                    print(
                        f"[ros_zed_stereo] rx left={counts['left']} right={counts['right']}",
                        flush=True,
                    )

        threading.Thread(target=_log_counts, daemon=True).start()

        self._node = Node("ros_zed_stereo_frames")
        self._node.create_subscription(
            CompressedImage, self._left_topic, on_left, qos_profile=qos
        )
        self._node.create_subscription(
            CompressedImage, self._right_topic, on_right, qos_profile=qos
        )
        print(
            f"[ros_zed_stereo] Subscribed: {self._left_topic}, {self._right_topic}",
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
