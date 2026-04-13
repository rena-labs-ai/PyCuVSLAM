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

RESOLUTION = (640, 360)  # (1280, 800)
FPS = 30
IR_EXPOSURE_US = 10000  # Manual exposure in µs for IR stereo
WARMUP_FRAMES = 500
IMAGE_JITTER_THRESHOLD_NS = ((1000 / FPS) + 2) * 1e6
RS_WAIT_FOR_FRAMES_MS = 5000
IMU_FREQUENCY_ACCEL = 100
IMU_FREQUENCY_GYRO = 200
IMU_JITTER_THRESHOLD_NS = 12 * 1e6

MULTI_CAM_CONFIG_FILE = (
    "./cuvslam_examples/" "cuvslam_examples/realsense/frame_agx_rig.yaml"
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

    img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
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
            print(
                "Warning: VSLAM track failed (world_from_rig or slam_pose is None)",
                flush=True,
            )
            return

        self.frames_fed += 1
        now = time.monotonic()
        if now - self._last_log_time >= 1.0:
            raw_str = ", ".join(
                f"cam_{i}: {n}" for i, n in enumerate(self._raw_images_num)
            )
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


class RosRealsenseStereoTracker(BaseTracker):
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
            raise TimeoutError(f"Did not receive CameraInfo for {missing} within 30 s")
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
            info_topics[f"cam_{i}_right"] = _image_topic_to_camera_info_topic(
                right_topic
            )

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
            raise TimeoutError(f"Did not receive CameraInfo for {missing} within 30 s")
        print(f"[ros_multicam] CameraInfo received for {self._num_cameras} cameras")

        camera_params: Dict[str, Dict] = {}
        for i, stereo_cam in enumerate(self._stereo_cameras):
            left_msg = collector.received[f"cam_{i}_left"]
            right_msg = collector.received[f"cam_{i}_right"]
            assert left_msg is not None and right_msg is not None
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
                    "extrinsics": stereo_cam["left_camera"]["transform"],
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
                CompressedImage,
                lt,
                partial(aggregator.on_compressed_msg, i * 2),
                qos_profile=qos,
            )
            self._node.create_subscription(
                CompressedImage,
                rt,
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


def _image_topic_to_camera_info_topic(image_topic: str) -> str:
    """Derive CameraInfo topic from image topic. E.g. /cam/infra1/image_rect_raw -> /cam/infra1/camera_info."""
    base = image_topic.replace("/compressed", "").rsplit("/", 1)[0]
    return f"{base}/camera_info"


class RosRealsenseRGBDTracker(BaseTracker):
    """RGBD tracking from RealSense ROS topics (compressed color + raw depth).

    Default topics follow the standard RealSense ROS2 wrapper conventions:
      color: {base}/color/image_raw/compressed
      depth: {base}/depth/image_rect_raw
    """

    DEFAULT_COLOR_SUFFIX = "color/image_raw/compressed"
    DEFAULT_DEPTH_SUFFIX = "depth/image_rect_raw"

    def __init__(
        self,
        topic_base: str,
        depth_scale: float = 0.001,
    ) -> None:
        base = topic_base.rstrip("/")
        self._color_topic = f"{base}/{self.DEFAULT_COLOR_SUFFIX}"
        self._depth_topic = f"{base}/{self.DEFAULT_DEPTH_SUFFIX}"
        self._depth_scale = depth_scale
        self._running = False

    def setup_camera_parameters(self) -> dict:
        import time
        from functools import partial

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        color_info_topic = _image_topic_to_camera_info_topic(self._color_topic)
        print(f"[ros_rs_rgbd] Waiting for CameraInfo on {color_info_topic} ...")

        collector = _CameraInfoCollector(["color"])
        node = Node("ros_rs_rgbd_camera_info")
        node.create_subscription(
            CameraInfo, color_info_topic, partial(collector.on_info, "color"), 10
        )

        deadline = time.time() + 30.0
        while not collector.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()

        if collector.received["color"] is None:
            raise TimeoutError("Did not receive color CameraInfo within 30 s")
        print("[ros_rs_rgbd] CameraInfo received")

        msg = collector.received["color"]
        return {
            "left": {
                "intrinsics": _make_intrinsics_like(
                    msg.k[0], msg.k[4], msg.k[2], msg.k[5],
                    msg.width, msg.height,
                ),
            },
        }

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
        import message_filters
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, Image

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
                    f"[ros_rs_rgbd] raw_color={raw_counts[0]}/s raw_depth={raw_counts[1]}/s "
                    f"fed={frames_fed[0]}/s",
                    flush=True,
                )
                frames_fed[0] = 0
                raw_counts[0] = 0
                raw_counts[1] = 0

        threading.Thread(target=_log_stats, daemon=True).start()

        def on_rgbd_pair(color_msg, depth_msg):
            ts = color_msg.header.stamp.sec * 1_000_000_000 + color_msg.header.stamp.nanosec

            color_image = _decode_compressed_image(bytes(color_msg.data))
            if color_image is None:
                print("[ros_rs_rgbd] Warning: color decode failed", flush=True)
                return

            depth_image = np.frombuffer(bytes(depth_msg.data), dtype=np.uint16).reshape(
                depth_msg.height, depth_msg.width
            )

            vo_pose_estimate, slam_pose = tracker.track(
                ts, images=[color_image], depths=[depth_image]
            )
            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[ros_rs_rgbd] Warning: track failed", flush=True)
                return

            frames_fed[0] += 1
            landmarks = [Landmark(lm.id, lm.coords) for lm in tracker.get_last_landmarks()]
            output_queue.put(
                TrackingResult(
                    ts,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose),
                    (color_image, depth_image),
                    landmarks,
                )
            )

        self._node = Node("ros_rs_rgbd_frames")
        color_sub = message_filters.Subscriber(
            self._node, CompressedImage, self._color_topic, qos_profile=qos
        )
        depth_sub = message_filters.Subscriber(
            self._node, Image, self._depth_topic, qos_profile=qos
        )
        color_sub.registerCallback(lambda _: raw_counts.__setitem__(0, raw_counts[0] + 1))
        depth_sub.registerCallback(lambda _: raw_counts.__setitem__(1, raw_counts[1] + 1))
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=RS_STEREO_SYNC_SLOP_SEC
        )
        self._sync.registerCallback(on_rgbd_pair)

        print(
            f"[ros_rs_rgbd] Subscribed (ApproximateTimeSynchronizer "
            f"slop={RS_STEREO_SYNC_SLOP_SEC * 1000:.0f}ms): "
            f"{self._color_topic}, {self._depth_topic}",
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
