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
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pyzed.sl as sl

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.camera_utils import (
    get_rs_imu,
    get_rs_stereo_rig,
    get_rs_vio_rig,
)
from cuvslam_examples.realsense.tracker import (
    BaseTracker,
    _CameraInfoCollector,
    _make_intrinsics_like,
    _spin_ros_node,
)
from cuvslam_examples.realsense.utils import Landmark, Pose

from cuvslam_examples.zed.camera_utils import get_zed_stereo_rig, setup_zed_camera

SLOP_SEC = 0.001  # 5 ms
RESOLUTION = (640, 480)
FPS = 60
WARMUP_FRAMES = 10
IMAGE_JITTER_THRESHOLD_NS = ((1000 / FPS) + 2) * 1e6

# ZED ROS2: camera_info is at {image_topic}/camera_info (e.g. .../color/rect/image/camera_info)
DEFAULT_ZED_LEFT_TOPIC = "/zed_base/zed_node/left/color/rect/image/compressed"
DEFAULT_ZED_RIGHT_TOPIC = "/zed_base/zed_node/right/color/rect/image/compressed"


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
        cfg.rectified_stereo_camera = not self._raw
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


def _collect_zed_stereo_camera_info(
    left_topic: str, right_topic: str, node_name: str, log_prefix: str
):
    """Spin a temporary ROS node to collect CameraInfo for left and right cameras.

    Returns (left_msg, right_msg) or raises TimeoutError.
    """
    import time as _time
    from functools import partial

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo

    left_info_topic = _zed_image_to_camera_info_topic(left_topic)
    right_info_topic = _zed_image_to_camera_info_topic(right_topic)
    print(
        f"[{log_prefix}] Waiting for CameraInfo on {left_info_topic}, {right_info_topic} ..."
    )

    collector = _CameraInfoCollector(["left", "right"])
    node = Node(node_name)
    node.create_subscription(
        CameraInfo, left_info_topic, partial(collector.on_info, "left"), 10
    )
    node.create_subscription(
        CameraInfo, right_info_topic, partial(collector.on_info, "right"), 10
    )

    deadline = _time.time() + 30.0
    while not collector.has_all() and _time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()

    missing = [k for k, v in collector.received.items() if v is None]
    if missing:
        raise TimeoutError(f"Did not receive CameraInfo for {missing} within 30 s")
    print(f"[{log_prefix}] CameraInfo received")

    left_msg = collector.received["left"]
    right_msg = collector.received["right"]
    assert left_msg is not None and right_msg is not None
    return left_msg, right_msg


def _build_zed_stereo_camera_params(left_msg, right_msg) -> Dict[str, Dict]:
    """Build the camera_params dict (left/right intrinsics + right extrinsics) from CameraInfo msgs."""
    # Baseline from right camera P matrix: P[0,3] = -fx*baseline
    fx_right = right_msg.p[0]
    baseline = -right_msg.p[3] / fx_right if fx_right != 0 else 0.12
    right_extrinsics = [[1, 0, 0, baseline], [0, 1, 0, 0], [0, 0, 1, 0]]
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
        left_msg, right_msg = _collect_zed_stereo_camera_info(
            self._left_topic,
            self._right_topic,
            node_name="ros_zed_stereo_camera_info",
            log_prefix="ros_zed_stereo",
        )
        return _build_zed_stereo_camera_params(left_msg, right_msg)

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=True,
            use_denoising=False,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_stereo_rig(camera_params)

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
        raw_counts = [0, 0]
        decode_sum_ms = [0.0]
        track_sum_ms = [0.0]

        def _log_stats():
            while self._running:
                time.sleep(1.0)
                n = frames_fed[0]
                d_avg = decode_sum_ms[0] / n if n else 0.0
                t_avg = track_sum_ms[0] / n if n else 0.0
                print(
                    f"[ros_zed_stereo] raw_cam_0={raw_counts[0]}/s raw_cam_1={raw_counts[1]}/s "
                    f"fed={n}/s avg_decode_ms={d_avg:.2f} avg_track_ms={t_avg:.2f}",
                    flush=True,
                )
                frames_fed[0] = 0
                raw_counts[0] = 0
                raw_counts[1] = 0
                decode_sum_ms[0] = 0.0
                track_sum_ms[0] = 0.0

        threading.Thread(target=_log_stats, daemon=True).start()

        last_ts = [0]

        def on_stereo_pair(left_msg, right_msg):
            ts = (
                left_msg.header.stamp.sec * 1_000_000_000
                + left_msg.header.stamp.nanosec
            )

            # Skip non-monotonic timestamps
            if ts <= last_ts[0]:
                return
            last_ts[0] = ts

            t_decode = time.monotonic()
            images = [
                _decode_zed_compressed_image(left_msg.data),
                _decode_zed_compressed_image(right_msg.data),
            ]
            decode_ms = (time.monotonic() - t_decode) * 1000

            if any(img is None for img in images):
                print("[ros_zed_stereo] Warning: decode failed", flush=True)
                return

            t0 = time.monotonic()
            vo_pose_estimate, slam_pose = tracker.track(ts, images)
            track_ms = (time.monotonic() - t0) * 1000

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[ros_zed_stereo] Warning: track failed", flush=True)
                return

            frames_fed[0] += 1
            decode_sum_ms[0] += decode_ms
            track_sum_ms[0] += track_ms

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

        self._node = Node("ros_zed_stereo_frames")
        left_sub = message_filters.Subscriber(
            self._node, CompressedImage, self._left_topic, qos_profile=qos
        )
        right_sub = message_filters.Subscriber(
            self._node, CompressedImage, self._right_topic, qos_profile=qos
        )
        left_sub.registerCallback(
            lambda _: raw_counts.__setitem__(0, raw_counts[0] + 1)
        )
        right_sub.registerCallback(
            lambda _: raw_counts.__setitem__(1, raw_counts[1] + 1)
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=10, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_stereo_pair)

        print(
            f"[ros_zed_stereo] Subscribed (ApproximateTimeSynchronizer slop={SLOP_SEC*1000:.0f}ms): "
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


DEFAULT_ZED_IMU_TOPIC = "/zed/zed_node/imu/data"
ZED_IMU_FREQUENCY = 200
IMU_JITTER_THRESHOLD_NS = 12 * 1e6


def _lookup_zed_imu_extrinsics(camera: str, timeout_sec: float = 5.0):
    """Look up cam_from_imu transform via TF for a given ZED camera model.

    Frame names are derived from the camera model, e.g. for 'zed2i':
      {camera}_base_left_camera_frame <- {camera}_base_imu_link

    Returns an object with .translation attribute compatible with rig_from_imu_pose().
    """
    import rclpy
    from rclpy.node import Node
    import tf2_ros

    target_frame = f"{camera}_base_left_camera_frame"
    source_frame = f"{camera}_base_imu_link"

    node = Node("zed_vio_tf_lookup")
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

    deadline = time.time() + timeout_sec
    transform = None
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
        try:
            transform = tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
            break
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            continue
    node.destroy_node()

    if transform is None:
        raise TimeoutError(
            f"Could not look up TF {source_frame} -> {target_frame} within {timeout_sec}s"
        )

    t = transform.transform.translation

    class _Extrinsics:
        pass

    ext = _Extrinsics()
    ext.translation = np.array([t.x, t.y, t.z])
    return ext


class RosZedVIOTracker(BaseTracker):
    """Visual-Inertial Odometry tracking from ZED ROS topics (stereo images + IMU)."""

    def __init__(
        self,
        left_topic: str = DEFAULT_ZED_LEFT_TOPIC,
        right_topic: str = DEFAULT_ZED_RIGHT_TOPIC,
        imu_topic: str = DEFAULT_ZED_IMU_TOPIC,
        camera: str = "zed2i",
    ) -> None:
        self._left_topic = left_topic
        self._right_topic = right_topic
        self._imu_topic = imu_topic
        self._camera = camera
        self._running = False

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        left_msg, right_msg = _collect_zed_stereo_camera_info(
            self._left_topic,
            self._right_topic,
            node_name="ros_zed_vio_camera_info",
            log_prefix="ros_zed_vio",
        )
        params = _build_zed_stereo_camera_params(left_msg, right_msg)

        print("[ros_zed_vio] Looking up IMU extrinsics from TF ...")
        cam_from_imu = _lookup_zed_imu_extrinsics(self._camera)
        print(f"[ros_zed_vio] cam_from_imu translation: {cam_from_imu.translation}")
        params["imu"] = {"cam_from_imu": cam_from_imu}
        return params

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            debug_imu_mode=False,
            odometry_mode=vslam.Tracker.OdometryMode.Inertial,
            rectified_stereo_camera=False,
            use_denoising=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        h = camera_params["left"]["intrinsics"].height
        return get_rs_vio_rig(camera_params, border_bottom=h // 4)

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=True, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import message_filters
        from sensor_msgs.msg import CompressedImage, Imu
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._running = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        imu_buffer: List[vslam.ImuMeasurement] = []
        last_imu_ts: List[Optional[int]] = [None]
        frames_fed = [0]
        imu_fed = [0]

        def _log_stats():
            while self._running:
                time.sleep(1.0)
                print(
                    f"[ros_zed_vio] fed={frames_fed[0]}/s imu={imu_fed[0]}/s",
                    flush=True,
                )
                frames_fed[0] = 0
                imu_fed[0] = 0

        threading.Thread(target=_log_stats, daemon=True).start()

        last_ts = [0]

        def _on_imu(msg: Imu) -> None:
            ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if last_imu_ts[0] is not None and ts_ns <= last_imu_ts[0]:
                return
            last_imu_ts[0] = ts_ns
            m = vslam.ImuMeasurement()
            m.timestamp_ns = ts_ns
            a = msg.linear_acceleration
            g = msg.angular_velocity
            m.linear_accelerations = [a.x, a.y, a.z]
            m.angular_velocities = [g.x, g.y, g.z]
            imu_buffer.append(m)

        def on_stereo_pair(left_msg, right_msg):
            ts = (
                left_msg.header.stamp.sec * 1_000_000_000
                + left_msg.header.stamp.nanosec
            )

            # Skip non-monotonic timestamps
            if ts <= last_ts[0]:
                return
            last_ts[0] = ts

            # Flush IMU measurements up to this image timestamp
            remaining = []
            for m in imu_buffer:
                if m.timestamp_ns <= ts:
                    tracker.register_imu_measurement(0, m)
                    imu_fed[0] += 1
                else:
                    remaining.append(m)
            imu_buffer[:] = remaining

            images = [
                _decode_zed_compressed_image(left_msg.data),
                _decode_zed_compressed_image(right_msg.data),
            ]
            if any(img is None for img in images):
                print("[ros_zed_vio] Warning: decode failed", flush=True)
                return

            vo_pose_estimate, slam_pose = tracker.track(ts, images)
            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[ros_zed_vio] Warning: track failed", flush=True)
                return

            frames_fed[0] += 1
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

        self._node = Node("ros_zed_vio_frames")
        left_sub = message_filters.Subscriber(
            self._node, CompressedImage, self._left_topic, qos_profile=qos
        )
        right_sub = message_filters.Subscriber(
            self._node, CompressedImage, self._right_topic, qos_profile=qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=10, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_stereo_pair)
        self._node.create_subscription(Imu, self._imu_topic, _on_imu, qos_profile=qos)

        print(
            f"[ros_zed_vio] Subscribed (ApproximateTimeSynchronizer slop={SLOP_SEC*1000:.0f}ms): "
            f"{self._left_topic}, {self._right_topic}, {self._imu_topic}",
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
