import queue
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import time


import numpy as np
import pyrealsense2 as rs
import yaml

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.camera_utils import (
    configure_device,
    get_camera_intrinsics,
    get_rs_multi_rig,
    get_rs_stereo_rig,
    get_rs_vio_rig,
    setup_pipeline,
    _make_intrinsics_like,
)
from cuvslam_examples.realsense.utils import Landmark, Pose

RESOLUTION = (640, 360) # (1280, 800)
FPS = 30
IR_EXPOSURE_US = 10000  # Manual exposure in µs for IR stereo
WARMUP_FRAMES = 500
IMAGE_JITTER_THRESHOLD_NS = ((1000 / FPS) + 2) * 1e6
RS_WAIT_FOR_FRAMES_MS = 5000
IMU_FREQUENCY_ACCEL = 100
IMU_FREQUENCY_GYRO = 200
IMU_JITTER_THRESHOLD_NS = 12 * 1e6

MULTI_CAM_CONFIG_FILE = (
    "./cuvslam_examples/"
    "cuvslam_examples/realsense/frame_agx_rig.yaml"
)
SYNC_MATCHING_THRESHOLD_NS = 100 * 1e6

ROS_INFRA1_TOPIC = "camera/infra1/image_rect_raw/compressed"
ROS_INFRA2_TOPIC = "camera/infra2/image_rect_raw/compressed"

# ApproximateTimeSynchronizer slop for ROS stereo pairs (aligned with ros_zed_stereo).
RS_STEREO_SYNC_SLOP_SEC = 0.001


class BaseTracker(ABC):
    """Abstract interface for cuVSLAM tracking strategies."""

    @property
    def num_viz_cameras(self) -> int:
        return 1

    @abstractmethod
    def setup_camera_parameters(self) -> dict: ...

    @abstractmethod
    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig: ...

    @abstractmethod
    def create_rig(self, camera_params: dict) -> vslam.Rig: ...

    @abstractmethod
    def create_slam_config(self) -> vslam.Tracker.SlamConfig: ...

    @abstractmethod
    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None: ...

    @abstractmethod
    def stop_streaming(self) -> None: ...

    def get_viz_image_indices(self) -> List[int]:
        """Image indices from TrackingResult.images to visualize."""
        return [0]

    def get_viz_observation_indices(self) -> List[int]:
        """Camera indices passed to get_last_observations for each viz camera."""
        return [0]


class StereoTracker(BaseTracker):
    """Stereo (IR-only) tracking strategy."""

    def __init__(self) -> None:
        self._pipeline: Optional[rs.pipeline] = None
        self._running = False

    def setup_camera_parameters(self) -> dict:
        config = rs.config()
        pipeline = rs.pipeline()


        config.enable_stream(
            rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
        )
        config.enable_stream(
            rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
        )

        profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        pipeline.stop()

        left_profile = frames[0].profile.as_video_stream_profile()
        right_profile = frames[1].profile.as_video_stream_profile()
        # print("left_to_right_extrinsics", right_profile.get_extrinsics_to(left_profile))

        return {
            "left": {"intrinsics": left_profile.intrinsics},
            "right": {
                "intrinsics": right_profile.intrinsics,
                "extrinsics": right_profile.get_extrinsics_to(left_profile),
            },
        }

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=True,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_stereo_rig(
            camera_params,
            border_bottom=RESOLUTION[1] // 3,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
        )
        config.enable_stream(
            rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
        )

        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        depth_sensor = device.query_sensors()[0]
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        if depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, IR_EXPOSURE_US)
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

        self._pipeline.start(config)
        self._running = True

        threading.Thread(
            target=self._spin,
            args=(tracker, output_queue),
            daemon=True,
        ).start()

    def stop_streaming(self) -> None:
        self._running = False
        if self._pipeline:
            self._pipeline.stop()

    def _spin(self, tracker: vslam.Tracker, output_queue: queue.Queue) -> None:
        frame_id = 0
        prev_timestamp: Optional[int] = None
        dropped_count = 0

        while self._running:
            frames = self._pipeline.wait_for_frames(RS_WAIT_FOR_FRAMES_MS)
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)

            if not left_frame or not right_frame:
                continue

            frame_id += 1
            timestamp = int(left_frame.timestamp * 1e6)

            if prev_timestamp is not None:
                timestamp_diff = timestamp - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                    dropped_count += 1
                    print(
                        f"Warning: Camera stream message drop: frame_id={frame_id} "
                        f"timestamp gap ({timestamp_diff/1e6:.2f} ms) exceeds "
                        f"threshold {IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms "
                        f"(total_dropped={dropped_count})",
                        flush=True,
                    )

            prev_timestamp = timestamp

            images = (
                np.asanyarray(left_frame.get_data()),
                np.asanyarray(right_frame.get_data()),
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
                    [],
                )
            )


class _TimestampTracker:
    def __init__(
        self, low_rate_threshold_ns: float, high_rate_threshold_ns: float
    ) -> None:
        self.prev_low_rate_timestamp: Optional[int] = None
        self.prev_high_rate_timestamp: Optional[int] = None
        self.low_rate_threshold_ns = low_rate_threshold_ns
        self.high_rate_threshold_ns = high_rate_threshold_ns
        self.last_low_rate_timestamp: Optional[int] = None


class VioTracker(BaseTracker):
    """Visual-Inertial Odometry tracking strategy."""

    def __init__(self) -> None:
        self._ir_pipe: Optional[rs.pipeline] = None
        self._motion_pipe: Optional[rs.pipeline] = None

    def setup_camera_parameters(self) -> dict:
        config = rs.config()
        pipeline = rs.pipeline()

        config.enable_stream(
            rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
        )
        config.enable_stream(
            rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
        )
        config.enable_stream(
            rs.stream.accel, rs.format.motion_xyz32f, IMU_FREQUENCY_ACCEL
        )
        config.enable_stream(
            rs.stream.gyro, rs.format.motion_xyz32f, IMU_FREQUENCY_GYRO
        )

        profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        pipeline.stop()

        return {
            "left": {
                "intrinsics": frames[0].profile.as_video_stream_profile().intrinsics,
            },
            "right": {
                "intrinsics": frames[1].profile.as_video_stream_profile().intrinsics,
                "extrinsics": frames[1].profile.get_extrinsics_to(frames[0].profile),
            },
            "imu": {
                "cam_from_imu": frames[2].profile.get_extrinsics_to(frames[0].profile),
            },
        }

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            debug_imu_mode=False,
            odometry_mode=vslam.Tracker.OdometryMode.Inertial,
            rectified_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_vio_rig(
            camera_params,
            border_bottom=RESOLUTION[1] // 3,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=True, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        self._ir_pipe = rs.pipeline()
        ir_config = rs.config()
        ir_config.enable_stream(
            rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
        )
        ir_config.enable_stream(
            rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
        )

        config_temp = rs.config()
        ir_wrapper = rs.pipeline_wrapper(self._ir_pipe)
        ir_profile = config_temp.resolve(ir_wrapper)
        device = ir_profile.get_device()

        depth_sensor = device.query_sensors()[0]
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        if depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, IR_EXPOSURE_US)
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

        self._motion_pipe = rs.pipeline()
        motion_config = rs.config()
        motion_config.enable_stream(
            rs.stream.accel, rs.format.motion_xyz32f, IMU_FREQUENCY_ACCEL
        )
        motion_config.enable_stream(
            rs.stream.gyro, rs.format.motion_xyz32f, IMU_FREQUENCY_GYRO
        )

        self._motion_pipe.start(motion_config)
        self._ir_pipe.start(ir_config)

        ts = _TimestampTracker(IMAGE_JITTER_THRESHOLD_NS, IMU_JITTER_THRESHOLD_NS)

        threading.Thread(
            target=self._imu_loop,
            args=(tracker, ts),
            daemon=True,
        ).start()

        threading.Thread(
            target=self._camera_loop,
            args=(tracker, output_queue, ts),
            daemon=True,
        ).start()

    def stop_streaming(self) -> None:
        if self._motion_pipe:
            self._motion_pipe.stop()
        if self._ir_pipe:
            self._ir_pipe.stop()

    def _imu_loop(self, tracker: vslam.Tracker, ts: _TimestampTracker) -> None:
        try:
            while True:
                imu_frames = self._motion_pipe.wait_for_frames()
                accel_frame = imu_frames.first_or_default(rs.stream.accel)
                gyro_frame = imu_frames.first_or_default(rs.stream.gyro)

                if not accel_frame or not gyro_frame:
                    continue

                current_timestamp = int(accel_frame.timestamp * 1e6)

                if (
                    ts.last_low_rate_timestamp is not None
                    and current_timestamp < ts.last_low_rate_timestamp
                ):
                    continue

                timestamp_diff = 0
                if ts.prev_high_rate_timestamp is not None:
                    timestamp_diff = current_timestamp - ts.prev_high_rate_timestamp
                    if timestamp_diff > ts.high_rate_threshold_ns:
                        print(
                            f"Warning: IMU stream message drop: timestamp gap "
                            f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                            f"{ts.high_rate_threshold_ns/1e6:.2f} ms"
                        )
                    elif timestamp_diff < 0:
                        print("Warning: IMU messages are not sequential")

                if timestamp_diff < 0:
                    continue

                ts.prev_high_rate_timestamp = deepcopy(current_timestamp)

                imu_measurement = vslam.ImuMeasurement()
                imu_measurement.timestamp_ns = current_timestamp

                accel_data = accel_frame.as_motion_frame().get_motion_data()
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()

                imu_measurement.linear_accelerations = [
                    accel_data.x,
                    accel_data.y,
                    accel_data.z,
                ]
                imu_measurement.angular_velocities = [
                    gyro_data.x,
                    gyro_data.y,
                    gyro_data.z,
                ]

                if timestamp_diff > 0:
                    tracker.register_imu_measurement(0, imu_measurement)
        except Exception as e:
            print(f"IMU thread error: {e}")
            import traceback

            traceback.print_exc()

    def _camera_loop(
        self,
        tracker: vslam.Tracker,
        output_queue: queue.Queue,
        ts: _TimestampTracker,
    ) -> None:
        try:
            while True:
                ir_frames = self._ir_pipe.wait_for_frames()
                ir_left_frame = ir_frames.get_infrared_frame(1)
                ir_right_frame = ir_frames.get_infrared_frame(2)
                current_timestamp = int(ir_left_frame.timestamp * 1e6)

                if ts.prev_low_rate_timestamp is not None:
                    timestamp_diff = current_timestamp - ts.prev_low_rate_timestamp
                    if timestamp_diff > ts.low_rate_threshold_ns:
                        print(
                            f"Warning: Camera stream message drop: timestamp gap "
                            f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                            f"{ts.low_rate_threshold_ns/1e6:.2f} ms"
                        )

                ts.prev_low_rate_timestamp = deepcopy(current_timestamp)

                images = (
                    np.asanyarray(ir_left_frame.get_data()),
                    np.asanyarray(ir_right_frame.get_data()),
                )

                odom_pose_estimate, slam_pose = tracker.track(current_timestamp, images)
                odom_pose = odom_pose_estimate.world_from_rig.pose

                output_queue.put(
                    TrackingResult(
                        current_timestamp,
                        Pose(odom_pose),
                        Pose(slam_pose),
                        images,
                    )
                )
                ts.last_low_rate_timestamp = current_timestamp
        except Exception as e:
            print(f"Camera thread error: {e}")


class MultiCameraTracker(BaseTracker):
    """Multi-camera stereo tracking strategy using hardware-synced RealSense rigs."""

    def __init__(self, config_file: str = MULTI_CAM_CONFIG_FILE) -> None:
        self._config_file = config_file
        self._pipelines: List[rs.pipeline] = []
        self._configs: List[rs.config] = []
        self._running = False
        self._stereo_cameras: List[Dict] = []
        self._jitter_lock = threading.Lock()
        self._jitter_points: List[tuple] = []

    @property
    def num_viz_cameras(self) -> int:
        return len(self._stereo_cameras)

    def get_viz_image_indices(self) -> List[int]:
        return list(range(0, len(self._stereo_cameras) * 2, 2))

    def get_viz_observation_indices(self) -> List[int]:
        return list(range(0, len(self._stereo_cameras) * 2, 2))

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        with open(self._config_file, "r") as f:
            config_data = yaml.safe_load(f)

        self._stereo_cameras = config_data["stereo_cameras"]
        serial_numbers = [cam["serial"] for cam in self._stereo_cameras]

        self._pipelines = []
        self._configs = []
        for serial in serial_numbers:
            pipeline, config = setup_pipeline(serial, fps=FPS)
            self._pipelines.append(pipeline)
            self._configs.append(config)

        camera_params: Dict[str, Dict] = {}
        for i, (pipeline, config, stereo_cam) in enumerate(
            zip(self._pipelines, self._configs, self._stereo_cameras)
        ):
            left_intrinsics, right_intrinsics = get_camera_intrinsics(pipeline, config)
            camera_params[f"camera_{i + 1}"] = {
                "left": {
                    "intrinsics": left_intrinsics,
                    "extrinsics": stereo_cam["left_camera"]["transform"],
                },
                "right": {
                    "intrinsics": right_intrinsics,
                    "extrinsics": stereo_cam["right_camera"]["transform"],
                },
            }

        return camera_params

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            rectified_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_multi_rig(
            camera_params,
            border_bottom=RESOLUTION[1] // 3,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def drain_jitter_points(self) -> List[tuple]:
        with self._jitter_lock:
            pts = list(self._jitter_points)
            self._jitter_points.clear()
            return pts

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        get_latest_fast_lio: Optional[Callable[[], Optional[object]]] = kwargs.get(
            "get_latest_fast_lio"
        )
        configure_device(self._pipelines[0], self._configs[0], is_master=True)
        for pipeline, config in zip(self._pipelines[1:], self._configs[1:]):
            configure_device(pipeline, config, is_master=False)

        for pipeline, config in zip(self._pipelines, self._configs):
            pipeline.start(config)

        self._running = True
        self._raw_queue: queue.Queue = queue.Queue(maxsize=4)
        threading.Thread(
            target=self._camera_loop,
            args=(get_latest_fast_lio,),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._tracker_loop,
            args=(tracker, output_queue),
            daemon=True,
        ).start()

    def stop_streaming(self) -> None:
        self._running = False
        for pipeline in self._pipelines:
            pipeline.stop()

    def _camera_loop(
        self, get_latest_fast_lio: Optional[Callable[[], Optional[object]]]
    ) -> None:
        frame_id = 0
        prev_timestamp: Optional[int] = None
        last_slam_xy: Optional[tuple] = None
        num_cameras = len(self._pipelines)

        while self._running:
            all_timestamps: List[int] = []
            all_images: List[np.ndarray] = []
            skip = False

            for i, pipeline in enumerate(self._pipelines):
                frames = pipeline.wait_for_frames()
                left_frame = frames.get_infrared_frame(1)
                right_frame = frames.get_infrared_frame(2)

                if not left_frame or not right_frame:
                    skip = True
                    break

                all_timestamps.append(int(left_frame.timestamp * 1e6))
                all_images.extend(
                    [
                        np.asanyarray(left_frame.get_data()),
                        np.asanyarray(right_frame.get_data()),
                    ]
                )

            frame_id += 1

            if skip or frame_id < WARMUP_FRAMES:
                continue

            if len(all_timestamps) < num_cameras:
                continue

            ts = all_timestamps[0]
            if prev_timestamp is not None:
                gap = ts - prev_timestamp
                if gap > IMAGE_JITTER_THRESHOLD_NS:
                    print(
                        f"Warning: [camera_loop] Raw frame jitter: gap "
                        f"({gap / 1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_NS / 1e6:.2f} ms",
                        flush=True,
                    )
            prev_timestamp = ts

            max_diff = max(
                abs(all_timestamps[i] - all_timestamps[j])
                for i in range(num_cameras)
                for j in range(i + 1, num_cameras)
            )
            if max_diff > SYNC_MATCHING_THRESHOLD_NS:
                print(
                    f"Warning: Sync mismatch {max_diff / 1e6:.2f} ms "
                    f"exceeds threshold {SYNC_MATCHING_THRESHOLD_NS / 1e6:.2f} ms",
                    flush=True,
                )

            synced_odom = None
            if get_latest_fast_lio is not None:
                synced_odom = get_latest_fast_lio()

            try:
                self._raw_queue.put_nowait(
                    (all_timestamps[0], tuple(all_images), synced_odom)
                )
            except queue.Full:
                pass

    def _tracker_loop(self, tracker: vslam.Tracker, output_queue: queue.Queue) -> None:
        prev_timestamp: Optional[int] = None
        last_slam_xy: Optional[tuple] = None
        num_cameras = len(self._pipelines)

        while self._running:
            try:
                raw = self._raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            backlog = self._raw_queue.qsize()
            if backlog > 0:
                print(f"[tracker_loop] raw_queue backlog={backlog}", flush=True)

            timestamp, all_images, synced_odom = raw

            if prev_timestamp is not None:
                gap = timestamp - prev_timestamp
                if gap > IMAGE_JITTER_THRESHOLD_NS:
                    pos = (
                        f" at cuvslam ({last_slam_xy[0]:.3f}, {last_slam_xy[1]:.3f})"
                        if last_slam_xy
                        else ""
                    )
                    print(
                        f"Warning: Raw frame jitter (camera/queue drop): gap "
                        f"({gap / 1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_NS / 1e6:.2f} ms{pos} "
                        "(SLAM track is not lagging)",
                        flush=True,
                    )
                    if last_slam_xy:
                        with self._jitter_lock:
                            self._jitter_points.append(last_slam_xy)

            prev_timestamp = timestamp

            vo_pose_estimate, slam_pose = tracker.track(timestamp, all_images)

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                continue

            rf = Pose(slam_pose).to_robot_frame()
            last_slam_xy = (rf.translation[0], rf.translation[1])

            output_queue.put(
                TrackingResult(
                    timestamp,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose),
                    all_images,
                    synced_odom=synced_odom,
                )
            )


def _decode_compressed_image(raw_bytes: bytes) -> Optional[np.ndarray]:
    import cv2

    img = cv2.imdecode(
        np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if img is None:
        return None
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return np.ascontiguousarray(img)


class _CameraInfoCollector:
    """Accumulates CameraInfo messages keyed by topic identifier."""

    def __init__(self, keys) -> None:
        self.received = {k: None for k in keys}

    def on_info(self, key: str, msg) -> None:
        if self.received[key] is None:
            self.received[key] = msg

    def has_all(self) -> bool:
        return all(v is not None for v in self.received.values())


class _FrameAggregator:
    """Collects raw compressed bytes per camera slot; when all slots are
    filled, pops and decodes them, then calls tracker.track()."""

    def __init__(
        self,
        num_slots: int,
        tracker: "vslam.Tracker",
        output_queue: queue.Queue,
        decode_fn: Optional[Callable[[bytes], Optional[np.ndarray]]] = None,
    ) -> None:
        self._num_slots = num_slots
        self._tracker = tracker
        self._output_queue = output_queue
        self._decode_fn = decode_fn or _decode_compressed_image
        self._queues = [queue.Queue(maxsize=1) for _ in range(num_slots)]
        self.frames_fed = 0
        self._raw_images_num = [0] * num_slots
        self._last_log_time = time.monotonic()

    def on_compressed_msg(self, slot: int, msg) -> None:
        ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        q = self._queues[slot]
        try:
            q.get_nowait()
        except queue.Empty:
            pass

        q.put_nowait((ts, bytes(msg.data)))
        self._raw_images_num[slot] += 1

        if any(q.empty() for q in self._queues):
            return

        pairs = [q.get_nowait() for q in self._queues]
        ts_primary = pairs[0][0]
        t_decode = time.monotonic()
        images = [self._decode_fn(raw) for _, raw in pairs]
        decode_ms = (time.monotonic() - t_decode) * 1000
        if any(img is None for img in images):
            print("Warning: Failed to decode image, slot:", slot, flush=True)
            return

        vo_pose_estimate, slam_pose = self._tracker.track(ts_primary, images)

        if vo_pose_estimate.world_from_rig is None or slam_pose is None:
            print("Warning: VSLAM track failed (world_from_rig or slam_pose is None)", flush=True)
            return

        self.frames_fed += 1
        now = time.monotonic()
        if now - self._last_log_time >= 1.0:
            raw_str = ", ".join(f"cam_{i}: {n}" for i, n in enumerate(self._raw_images_num))
            print(f"[ros_multicam] fed={self.frames_fed}/s raw {raw_str}", flush=True)
            self.frames_fed = 0
            self._raw_images_num = [0] * self._num_slots
            self._last_log_time = now

        self._output_queue.put(
            TrackingResult(
                ts_primary,
                Pose(vo_pose_estimate.world_from_rig.pose),
                Pose(slam_pose),
                images,
            )
        )


def _spin_ros_node(node, tracker) -> None:
    import rclpy

    while tracker._running and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)


class RosRealSenseStereoTracker(BaseTracker):
    """Single RealSense D4xx stereo over ROS (infra1/infra2 compressed).

    Uses ApproximateTimeSynchronizer like RosZedStereoTracker: only time-aligned
    pairs (header stamps within slop) are decoded and fed to SLAM.
    """

    def __init__(self, topic_base: str) -> None:
        base = topic_base.rstrip("/")
        self._left_topic = f"{base}/{ROS_INFRA1_TOPIC}"
        self._right_topic = f"{base}/{ROS_INFRA2_TOPIC}"
        self._running = False

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        import time
        from functools import partial

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        left_info_topic = _image_topic_to_camera_info_topic(self._left_topic)
        right_info_topic = _image_topic_to_camera_info_topic(self._right_topic)
        print(
            f"[ros_realsense_stereo] Waiting for CameraInfo on "
            f"{left_info_topic}, {right_info_topic} ..."
        )

        collector = _CameraInfoCollector(["left", "right"])
        node = Node("ros_realsense_stereo_camera_info")
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
            raise TimeoutError(
                f"Did not receive CameraInfo for {missing} within 30 s"
            )
        print("[ros_realsense_stereo] CameraInfo received")

        left_msg = collector.received["left"]
        right_msg = collector.received["right"]
        assert left_msg is not None and right_msg is not None

        fx_right = right_msg.p[0]
        baseline = -right_msg.p[3] / fx_right if fx_right != 0 else 0.05
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
            rectified_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_stereo_rig(
            camera_params,
            border_bottom=RESOLUTION[1] // 3,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import message_filters
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage

        self._running = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        frames_fed = [0]
        raw_counts = [0, 0]

        def _log_stats():
            while self._running:
                time.sleep(1.0)
                print(
                    f"[ros_realsense_stereo] raw_cam_0={raw_counts[0]}/s "
                    f"raw_cam_1={raw_counts[1]}/s fed={frames_fed[0]}/s",
                    flush=True,
                )
                frames_fed[0] = 0
                raw_counts[0] = 0
                raw_counts[1] = 0

        threading.Thread(target=_log_stats, daemon=True).start()

        def on_stereo_pair(left_msg, right_msg):
            ts = (
                left_msg.header.stamp.sec * 1_000_000_000
                + left_msg.header.stamp.nanosec
            )

            images = [
                _decode_compressed_image(bytes(left_msg.data)),
                _decode_compressed_image(bytes(right_msg.data)),
            ]

            if any(img is None for img in images):
                print("[ros_realsense_stereo] Warning: decode failed", flush=True)
                return

            vo_pose_estimate, slam_pose = tracker.track(ts, images)

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[ros_realsense_stereo] Warning: track failed", flush=True)
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

        self._node = Node("ros_realsense_stereo_frames")
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
            [left_sub, right_sub],
            queue_size=10,
            slop=RS_STEREO_SYNC_SLOP_SEC,
        )
        self._sync.registerCallback(on_stereo_pair)

        print(
            f"[ros_realsense_stereo] Subscribed (ApproximateTimeSynchronizer "
            f"slop={RS_STEREO_SYNC_SLOP_SEC * 1000:.0f}ms): "
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


def _camera_topics_from_config(config_file: str) -> List[Tuple[str, str]]:
    """Derive (left, right) topic pairs from rig config. Uses name for /{name}/camera/..."""
    with open(config_file) as f:
        data = yaml.safe_load(f)
    topics = []
    for c in data["stereo_cameras"]:
        name = str(c.get("name", c["serial"])).strip("/")
        left = f"/{name}/{ROS_INFRA1_TOPIC}"
        right = f"/{name}/{ROS_INFRA2_TOPIC}"
        topics.append((left, right))
    return topics


class RosMulticamTracker(BaseTracker):
    """Multi-camera stereo tracking from live ROS compressed topics.

    Each camera topic callback stores the latest decoded image in a shared
    slot. When all slots are filled, images are popped and passed to
    tracker.track(). Camera topics are derived from config (stereo_cameras with name).
    """

    def __init__(self, config_file: str) -> None:
        self._config_file = config_file
        self._camera_topics = _camera_topics_from_config(config_file)
        self._num_cameras = len(self._camera_topics)
        self._stereo_cameras: List[Dict] = []
        self._running = False

    @property
    def num_viz_cameras(self) -> int:
        return self._num_cameras

    def get_viz_image_indices(self) -> List[int]:
        return list(range(0, self._num_cameras * 2, 2))

    def get_viz_observation_indices(self) -> List[int]:
        return list(range(0, self._num_cameras * 2, 2))

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        import time
        from functools import partial

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        with open(self._config_file, "r") as f:
            config_data = yaml.safe_load(f)
        self._stereo_cameras = config_data["stereo_cameras"]

        if len(self._stereo_cameras) != self._num_cameras:
            raise ValueError(
                f"Config has {len(self._stereo_cameras)} stereo cameras "
                f"but {self._num_cameras} topic pairs were provided"
            )

        info_topics: Dict[str, str] = {}
        for i, (left_topic, right_topic) in enumerate(self._camera_topics):
            info_topics[f"cam_{i}_left"] = _image_topic_to_camera_info_topic(left_topic)
            info_topics[f"cam_{i}_right"] = _image_topic_to_camera_info_topic(right_topic)

        print(f"[ros_multicam] Waiting for CameraInfo on {len(info_topics)} topics ...")
        for key, topic in info_topics.items():
            print(f"  {key}: {topic}")

        collector = _CameraInfoCollector(info_topics.keys())
        node = Node("ros_multicam_camera_info")
        for key, topic in info_topics.items():
            node.create_subscription(
                CameraInfo, topic, partial(collector.on_info, key), 10
            )

        deadline = time.time() + 30.0
        while not collector.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()

        missing = [k for k, v in collector.received.items() if v is None]
        if missing:
            raise TimeoutError(
                f"Did not receive CameraInfo for {missing} within 30 s"
            )
        print(f"[ros_multicam] CameraInfo received for {self._num_cameras} cameras")

        camera_params: Dict[str, Dict] = {}
        for i, stereo_cam in enumerate(self._stereo_cameras):
            left_msg = collector.received[f"cam_{i}_left"]
            right_msg = collector.received[f"cam_{i}_right"]
            assert left_msg is not None and right_msg is not None
            camera_params[f"camera_{i + 1}"] = {
                "left": {
                    "intrinsics": _make_intrinsics_like(
                        left_msg.k[0], left_msg.k[4],
                        left_msg.k[2], left_msg.k[5],
                        left_msg.width, left_msg.height,
                    ),
                    "extrinsics": stereo_cam["left_camera"]["transform"],
                },
                "right": {
                    "intrinsics": _make_intrinsics_like(
                        right_msg.k[0], right_msg.k[4],
                        right_msg.k[2], right_msg.k[5],
                        right_msg.width, right_msg.height,
                    ),
                    "extrinsics": stereo_cam["right_camera"]["transform"],
                },
            }
        return camera_params

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_multi_rig(
            camera_params,
            border_bottom=RESOLUTION[1] // 4,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        from functools import partial

        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage

        self._running = True
        num_slots = self._num_cameras * 2

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        aggregator = _FrameAggregator(num_slots, tracker, output_queue)
        self._node = Node("ros_multicam_frames")
        for i, (lt, rt) in enumerate(self._camera_topics):
            self._node.create_subscription(
                CompressedImage, lt,
                partial(aggregator.on_compressed_msg, i * 2),
                qos_profile=qos,
            )
            self._node.create_subscription(
                CompressedImage, rt,
                partial(aggregator.on_compressed_msg, i * 2 + 1),
                qos_profile=qos,
            )
            print(f"[ros_multicam] Subscribed: {lt}, {rt}")

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


BAG_SYNC_THRESHOLD_NS = 100 * 1e6  # 100 ms (network/robot sync can exceed 50 ms)
DEFAULT_BAG_CONFIG_FILE = str(Path(__file__).parent / "frame_agx_rig.yaml")


def _stereo_topic_base(cam: Dict, prefix: str) -> str:
    """Resolve ROS topic base for a stereo camera from config entry.
    Supports: topic_base (explicit), serial (RealSense), camera (Galileo-style).
    """
    if "topic_base" in cam:
        base = str(cam["topic_base"]).strip("/")
        return f"/{base}" if base else ""
    if "serial" in cam:
        p = prefix.strip("/")
        return f"/{p}/serial_{cam['serial']}" if p else f"/serial_{cam['serial']}"
    if "camera" in cam:
        return f"/{cam['camera']}_stereo_camera"
    raise ValueError(
        f"stereo_cameras entry must have 'serial', 'camera', or 'topic_base': {cam}"
    )


def _stereo_camera_id(cam: Dict) -> str:
    """Unique id for filtering: serial or camera name."""
    if "serial" in cam:
        return cam["serial"]
    if "camera" in cam:
        return cam["camera"]
    if "topic_base" in cam:
        return str(cam["topic_base"])
    raise ValueError(f"stereo_cameras entry must have 'serial' or 'camera': {cam}")


class MultiCamBagTracker(BaseTracker):
    """Multi-camera stereo tracking from ROS bag (no live cameras).

    Subscribes to IR image topics and ground-truth Odometry. Uses intrinsics
    from /camera_info and extrinsics from YAML config.
    Supports N stereo cameras from rig config. Config may use 'serial' (RealSense)
    or 'camera' (Galileo-style) to identify each stereo pair.
    """

    def __init__(
        self,
        config_file: str = DEFAULT_BAG_CONFIG_FILE,
        serial_numbers: Optional[List[str]] = None,
        camera_names: Optional[List[str]] = None,
        camera_topic_prefix: str = "camera",
        left_ir_topic: str = "infra1/image_rect_raw/compressed",
        right_ir_topic: str = "infra2/image_rect_raw/compressed",
        ground_truth_topic: str = "/Odometry",
        sync_slop: float = 0.2,
    ) -> None:
        self._config_file = config_file
        self._serial_numbers = serial_numbers
        self._camera_names = camera_names
        self._camera_topic_prefix = camera_topic_prefix
        self._left_ir_topic = left_ir_topic
        self._right_ir_topic = right_ir_topic
        self._ground_truth_topic = ground_truth_topic
        self._sync_slop = sync_slop
        self._stereo_cameras: List[Dict] = []
        self._num_cameras = 0
        self._running = False
        self._ground_truth: List[tuple] = []  # (timestamp_ns, x, y, z)

    @property
    def num_viz_cameras(self) -> int:
        return self._num_cameras

    def get_viz_image_indices(self) -> List[int]:
        return list(range(0, self._num_cameras * 2, 2))

    def get_viz_observation_indices(self) -> List[int]:
        return list(range(0, self._num_cameras * 2, 2))

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        with open(self._config_file, "r") as f:
            config_data = yaml.safe_load(f)

        all_stereo = config_data["stereo_cameras"]
        id_to_cam = {_stereo_camera_id(c): c for c in all_stereo}
        filter_ids = self._serial_numbers or self._camera_names
        if filter_ids is not None:
            self._stereo_cameras = [id_to_cam[i] for i in filter_ids if i in id_to_cam]
            if len(self._stereo_cameras) != len(filter_ids):
                missing = set(filter_ids) - {
                    _stereo_camera_id(c) for c in self._stereo_cameras
                }
                raise ValueError(f"Cameras {missing} not found in {self._config_file}")
        else:
            self._stereo_cameras = all_stereo

        self._num_cameras = len(self._stereo_cameras)
        prefix = self._camera_topic_prefix.strip("/")
        self._camera_topic_bases = [
            _stereo_topic_base(c, prefix) for c in self._stereo_cameras
        ]

        def _topic_base(i: int) -> str:
            return self._camera_topic_bases[i]

        camera_info_msgs: Dict[str, Optional[CameraInfo]] = {}
        for i in range(self._num_cameras):
            base = _topic_base(i)
            camera_info_msgs[f"{base}/left"] = None
            camera_info_msgs[f"{base}/right"] = None

        def image_topic_to_camera_info(image_topic: str) -> str:
            # infra1/image_rect_raw or infra1/image_rect_raw/compressed -> infra1/camera_info
            parts = image_topic.split("/")
            stream = parts[0] if parts else "infra1"
            return f"{stream}/camera_info"

        left_info_topic = image_topic_to_camera_info(self._left_ir_topic)
        right_info_topic = image_topic_to_camera_info(self._right_ir_topic)

        class CameraInfoCollector(Node):
            def __init__(self, topics: Dict[str, str]):
                super().__init__("camera_info_collector")
                self._received: Dict[str, Optional[CameraInfo]] = {
                    k: None for k in topics
                }
                for key, topic in topics.items():
                    self.create_subscription(
                        CameraInfo, topic, lambda msg, k=key: self._cb(msg, k), 10
                    )

            def _cb(self, msg: CameraInfo, key: str) -> None:
                if self._received[key] is None:
                    self._received[key] = msg

            def has_all(self) -> bool:
                return all(v is not None for v in self._received.values())

            def get_all(self) -> Dict[str, CameraInfo]:
                return self._received

        topic_map = {}
        for i in range(self._num_cameras):
            base = _topic_base(i)
            topic_map[f"{base}/left"] = f"{base}/{left_info_topic}"
            topic_map[f"{base}/right"] = f"{base}/{right_info_topic}"

        print(f"[multicam_bag] Waiting for CameraInfo on {len(topic_map)} topics...")
        node = CameraInfoCollector(topic_map)
        import time

        deadline = time.time() + 30.0
        while not node.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)

        msgs = node.get_all()
        node.destroy_node()

        if not all(v is not None for v in msgs.values()):
            missing = [k for k, v in msgs.items() if v is None]
            raise TimeoutError(f"Did not receive CameraInfo for: {missing} within 30s")
        print(f"[multicam_bag] CameraInfo received for {self._num_cameras} cameras")

        camera_params: Dict[str, Dict] = {}
        for i, stereo_cam in enumerate(self._stereo_cameras):
            base = _topic_base(i)
            left_msg = msgs[f"{base}/left"]
            right_msg = msgs[f"{base}/right"]
            assert left_msg is not None and right_msg is not None

            left_intrinsics = _make_intrinsics_like(
                left_msg.k[0],
                left_msg.k[4],
                left_msg.k[2],
                left_msg.k[5],
                left_msg.width,
                left_msg.height,
            )
            right_intrinsics = _make_intrinsics_like(
                right_msg.k[0],
                right_msg.k[4],
                right_msg.k[2],
                right_msg.k[5],
                right_msg.width,
                right_msg.height,
            )
            camera_params[f"camera_{i + 1}"] = {
                "left": {
                    "intrinsics": left_intrinsics,
                    "extrinsics": stereo_cam["left_camera"]["transform"],
                },
                "right": {
                    "intrinsics": right_intrinsics,
                    "extrinsics": stereo_cam["right_camera"]["transform"],
                },
            }

        return camera_params

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            rectified_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_multi_rig(
            camera_params,
            border_bottom=360 // 3,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image, CompressedImage
        from nav_msgs.msg import Odometry
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        import cv2

        self._running = True
        self._ground_truth = []

        class BagSubscriberNode(Node):
            def __init__(inner_self, outer: "MultiCamBagTracker"):
                super().__init__("multicam_bag_subscriber")
                inner_self._outer = outer
                inner_self._use_compressed = "/compressed" in outer._left_ir_topic
                inner_self._gt_lock = threading.Lock()

                qos = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10,
                )

                img_msg_type = CompressedImage if inner_self._use_compressed else Image
                subs = []
                for i in range(outer._num_cameras):
                    base = outer._camera_topic_bases[i]
                    left_topic = f"{base}/{outer._left_ir_topic}"
                    right_topic = f"{base}/{outer._right_ir_topic}"
                    subs.append(
                        Subscriber(
                            inner_self, img_msg_type, left_topic, qos_profile=qos
                        )
                    )
                    subs.append(
                        Subscriber(
                            inner_self, img_msg_type, right_topic, qos_profile=qos
                        )
                    )
                    print(f"[multicam_bag] Subscribed: {left_topic}, {right_topic}")

                inner_self._frame_count = 0
                inner_self._img_counts = [0] * (outer._num_cameras * 2)
                print(
                    f"[multicam_bag] Sync slop={outer._sync_slop}s (increase with --sync-slop if no frames)"
                )
                inner_self._sync = ApproximateTimeSynchronizer(
                    subs, queue_size=30, slop=outer._sync_slop
                )
                inner_self._sync.registerCallback(inner_self._on_synced_images)

                def make_count_cb(idx):
                    def _cb(msg):
                        inner_self._img_counts[idx] += 1

                    return _cb

                for i in range(outer._num_cameras):
                    base = outer._camera_topic_bases[i]
                    lt = f"{base}/{outer._left_ir_topic}"
                    rt = f"{base}/{outer._right_ir_topic}"
                    inner_self.create_subscription(
                        img_msg_type, lt, make_count_cb(i * 2), qos_profile=qos
                    )
                    inner_self.create_subscription(
                        img_msg_type, rt, make_count_cb(i * 2 + 1), qos_profile=qos
                    )

                inner_self.create_subscription(
                    Odometry, outer._ground_truth_topic, inner_self._on_odom, 10
                )
                inner_self._tracker = tracker
                inner_self._output_queue = output_queue

            def _on_odom(inner_self, msg: Odometry) -> None:
                ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                p = msg.pose.pose.position
                with inner_self._gt_lock:
                    inner_self._outer._ground_truth.append((ts, p.x, p.y, p.z))

            def _on_synced_images(inner_self, *msgs) -> None:
                if not inner_self._outer._running:
                    return

                timestamps = []
                images = []
                for msg in msgs:
                    ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    timestamps.append(ts)
                    if inner_self._use_compressed:
                        buf = np.array(msg.data, dtype=np.uint8)
                        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                        if img is None:
                            return
                    else:
                        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                            msg.height, msg.step
                        )[:, : msg.width]
                    if len(img.shape) == 3:
                        img = img[:, :, 0]
                    images.append(np.asarray(img))

                max_diff = max(
                    abs(timestamps[i] - timestamps[j])
                    for i in range(len(timestamps))
                    for j in range(i + 1, len(timestamps))
                )
                if max_diff > BAG_SYNC_THRESHOLD_NS:
                    print(
                        f"Warning: Sync mismatch {max_diff / 1e6:.2f} ms "
                        f"exceeds threshold {BAG_SYNC_THRESHOLD_NS / 1e6:.2f} ms"
                    )

                primary_ts = timestamps[0]
                inner_self._frame_count += 1
                print(
                    f"[multicam_bag] Sending {len(images)} images to VSLAM (frame {inner_self._frame_count}, ts={primary_ts})"
                )
                vo_pose_estimate, slam_pose = inner_self._tracker.track(
                    primary_ts, tuple(images)
                )

                if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                    print(
                        f"[multicam_bag] VSLAM track failed (world_from_rig or slam_pose is None)"
                    )
                    return

                if inner_self._frame_count == 1:
                    print("[multicam_bag] First synced frame tracked")
                elif inner_self._frame_count % 100 == 0:
                    print(f"[multicam_bag] Processed {inner_self._frame_count} frames")

                inner_self._output_queue.put(
                    TrackingResult(
                        primary_ts,
                        Pose(vo_pose_estimate.world_from_rig.pose),
                        Pose(slam_pose),
                        tuple(images),
                    )
                )

        try:
            node = BagSubscriberNode(self)
        except Exception as e:
            self._running = False
            raise e

        def spin():
            try:
                import rclpy

                while self._running and rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.1)
            except Exception as e:
                print(f"Bag subscriber error: {e}")

        self._node = node
        self._spin_thread = threading.Thread(target=spin, daemon=True)
        self._spin_thread.start()

    def stop_streaming(self) -> None:
        self._running = False
        if hasattr(self, "_spin_thread"):
            self._spin_thread.join(timeout=2.0)
        if hasattr(self, "_node") and hasattr(self._node, "_img_counts"):
            print(
                f"[multicam_bag] Image msgs received per topic: {self._node._img_counts} "
                f"(sync slop={self._sync_slop}s, synced frames={getattr(self._node, '_frame_count', 0)})"
            )
        if hasattr(self, "_node"):
            self._node.destroy_node()

    def get_ground_truth_trajectory(self) -> List[tuple]:
        """Return [(timestamp_ns, x, y, z), ...] for ground truth."""
        return list(self._ground_truth)


def _image_topic_to_camera_info_topic(image_topic: str) -> str:
    """Derive CameraInfo topic from image topic. E.g. /cam/infra1/image_rect_raw -> /cam/infra1/camera_info."""
    base = image_topic.replace("/compressed", "").rsplit("/", 1)[0]
    return f"{base}/camera_info"


class RGBDTracker(BaseTracker):
    """RGBD (color + depth) tracking strategy."""

    def __init__(self) -> None:
        self._pipeline: Optional[rs.pipeline] = None
        self._running = False
        self._depth_scale: float = 0.0

    @property
    def num_viz_cameras(self) -> int:
        return 2

    def get_viz_image_indices(self) -> List[int]:
        return [0, 1]

    def get_viz_observation_indices(self) -> List[int]:
        return [0, 0]

    def setup_camera_parameters(self) -> dict:
        config = rs.config()
        pipeline = rs.pipeline()

        config.enable_stream(
            rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.rgb8, FPS
        )
        config.enable_stream(
            rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS
        )

        profile = pipeline.start(config)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"[camera] Depth scale: {self._depth_scale}")

        align = rs.align(rs.stream.color)
        frames = align.process(pipeline.wait_for_frames())
        color_intrinsics = (
            frames.get_color_frame().profile.as_video_stream_profile().intrinsics
        )
        pipeline.stop()

        return {"left": {"intrinsics": color_intrinsics}}

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        rgbd_settings = vslam.Tracker.OdometryRGBDSettings()
        rgbd_settings.depth_scale_factor = 1 / self._depth_scale
        rgbd_settings.depth_camera_id = 0
        rgbd_settings.enable_depth_stereo_tracking = False

        return vslam.Tracker.OdometryConfig(
            async_sba=True,
            enable_final_landmarks_export=True,
            odometry_mode=vslam.Tracker.OdometryMode.RGBD,
            rgbd_settings=rgbd_settings,
            rectified_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_stereo_rig(
            camera_params,
            border_bottom=RESOLUTION[1] // 3,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.rgb8, FPS
        )
        config.enable_stream(
            rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS
        )

        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        depth_sensor = device.query_sensors()[0]
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        if depth_sensor.supports(rs.option.inter_cam_sync_mode):
            depth_sensor.set_option(rs.option.inter_cam_sync_mode, 1)
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

        color_sensor = device.query_sensors()[1]
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        self._pipeline.start(config)
        self._running = True

        threading.Thread(
            target=self._spin,
            args=(tracker, output_queue),
            daemon=True,
        ).start()

    def stop_streaming(self) -> None:
        self._running = False
        if self._pipeline:
            self._pipeline.stop()

    def _spin(self, tracker: vslam.Tracker, output_queue: queue.Queue) -> None:
        frame_id = 0
        prev_timestamp: Optional[int] = None
        jitter_threshold = ((1000 / FPS) + 2) * 1e6
        align = rs.align(rs.stream.color)

        while self._running:
            frames = self._pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame_id += 1
            timestamp = int(color_frame.timestamp * 1e6)

            if prev_timestamp is not None:
                gap = timestamp - prev_timestamp
                if gap > jitter_threshold:
                    print(
                        f"Warning: Camera stream drop: gap "
                        f"({gap / 1e6:.2f} ms) exceeds threshold "
                        f"{jitter_threshold / 1e6:.2f} ms"
                    )

            prev_timestamp = timestamp

            if frame_id <= WARMUP_FRAMES:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            vo_pose_estimate, slam_pose = tracker.track(
                timestamp, images=[color_image], depths=[depth_image]
            )

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                continue

            output_queue.put(
                TrackingResult(
                    timestamp,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose),
                    (color_image, depth_image),
                )
            )
