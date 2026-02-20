import queue
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import numpy as np
import pyrealsense2 as rs

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.camera_utils import get_rs_stereo_rig, get_rs_vio_rig
from cuvslam_examples.realsense.utils import Landmark, Pose

RESOLUTION = (640, 360)
FPS = 15
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_NS = ((1000 / FPS) + 2) * 1e6
IMU_FREQUENCY_ACCEL = 100
IMU_FREQUENCY_GYRO = 200
IMU_JITTER_THRESHOLD_NS = 12 * 1e6


class BaseTracker(ABC):
    """Abstract interface for cuVSLAM tracking strategies."""

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
        self, tracker: vslam.Tracker, output_queue: queue.Queue
    ) -> None: ...

    @abstractmethod
    def stop_streaming(self) -> None: ...


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

        return {
            "left": {"intrinsics": left_profile.intrinsics},
            "right": {
                "intrinsics": right_profile.intrinsics,
                "extrinsics": right_profile.get_extrinsics_to(left_profile),
            },
        }

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            horizontal_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_stereo_rig(camera_params)

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue
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

        while self._running:
            frames = self._pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)

            if not left_frame or not right_frame:
                continue

            frame_id += 1
            timestamp = int(left_frame.timestamp * 1e6)

            if prev_timestamp is not None:
                timestamp_diff = timestamp - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms"
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
                    landmarks,
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
            horizontal_stereo_camera=True,
        )

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        return get_rs_vio_rig(camera_params)

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue
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
                    TrackingResult(current_timestamp, odom_pose, slam_pose, images)
                )
                ts.last_low_rate_timestamp = current_timestamp
        except Exception as e:
            print(f"Camera thread error: {e}")
