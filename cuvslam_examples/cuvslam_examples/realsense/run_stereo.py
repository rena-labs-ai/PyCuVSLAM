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
from typing import List, Optional

import numpy as np
import pyrealsense2 as rs

import cuvslam as vslam
from camera_utils import get_rs_stereo_rig
from utils import OdomLogger, Pose
from visualizer import RerunVisualizer

# Constants
RESOLUTION = (640, 360)
FPS = 30
IR_EXPOSURE_US = 10000  # Manual exposure in µs (3300 default; higher = brighter)
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds


def main() -> None:
    """Main function for stereo tracking."""
    # Initialize RealSense configuration
    config = rs.config()
    pipeline = rs.pipeline()

    # Configure streams
    config.enable_stream(
        rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    config.enable_stream(
        rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )

    # Start pipeline to get intrinsics and extrinsics
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    pipeline.stop()

    # Prepare camera parameters
    camera_params = {'left': {}, 'right': {}}

    # Get extrinsics and intrinsics
    left_profile = frames[0].profile.as_video_stream_profile()
    right_profile = frames[1].profile.as_video_stream_profile()
    
    camera_params['left']['intrinsics'] = left_profile.intrinsics
    camera_params['right']['intrinsics'] = right_profile.intrinsics
    camera_params['right']['extrinsics'] = right_profile.get_extrinsics_to(
        left_profile
    )

    # Configure tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        horizontal_stereo_camera=True
    )

    # Create rig using utility function
    # Mask bottom third to exclude ground/robot body from feature detection
    rig = get_rs_stereo_rig(
        camera_params,
        border_bottom=RESOLUTION[1] // 3,
    )

    slam_config = vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)
    tracker = vslam.Tracker(rig, cfg, slam_config)
    odom_logger = OdomLogger()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Disable IR emitter if supported
    depth_sensor = device.query_sensors()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    # Set manual exposure (disables auto-exposure)
    if depth_sensor.supports(rs.option.exposure):
        depth_sensor.set_option(rs.option.exposure, IR_EXPOSURE_US)
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

    # Log exposure status
    if depth_sensor.supports(rs.option.exposure):
        exposure_us = int(depth_sensor.get_option(rs.option.exposure))
        print(f"[camera] exposure: {exposure_us} µs")

    visualizer = RerunVisualizer()

    # Start pipeline for tracking
    profile = pipeline.start(config)

    frame_id = 0
    prev_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()

            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            
            if not left_frame or not right_frame:
                print("Warning: Failed to get frames")
                continue

            frame_id += 1
            timestamp = int(left_frame.timestamp * 1e6)  # Convert to nanoseconds

            # Check timestamp difference with previous frame
            if prev_timestamp is not None:
                timestamp_diff = timestamp - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_MS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_MS/1e6:.2f} ms"
                    )

            # Store current timestamp for next iteration
            prev_timestamp = timestamp

            images = (
                np.asanyarray(left_frame.get_data()),
                np.asanyarray(right_frame.get_data())
            )

            # Warmup for specified number of frames
            if frame_id > WARMUP_FRAMES:
                vo_pose_estimate, slam_pose = tracker.track(timestamp, images)
                if vo_pose_estimate.world_from_rig is None:
                    print("Warning: Pose tracking not valid")
                    continue
                trajectory.append(vo_pose_estimate.world_from_rig.pose.translation)
                odom_logger.log(
                    frame_id,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose) if slam_pose is not None else None,
                )

                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=images,
                    observations_main_cam=[tracker.get_last_observations(0)],
                    slam_pose=slam_pose,
                    slam_trajectory=trajectory,
                    timestamp=timestamp,
                    gravity=None,
                    landmarks=None,
                    final_landmarks=None,
                    loop_closure_poses=None
                )

    finally:
        odom_logger.close()
        pipeline.stop()


if __name__ == "__main__":
    main()
