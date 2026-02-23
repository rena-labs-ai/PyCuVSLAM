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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs
import yaml
from copy import deepcopy

import cuvslam as vslam
from camera_utils import (
    configure_device,
    get_camera_intrinsics,
    get_rs_multi_rig,
    setup_pipeline,
)
from utils import OdomLogger, Pose
from visualizer import RerunVisualizer

# Constants
CONFIG_FILE = "src/rena_dependencies/pycuvslam/cuvslam_examples/cuvslam_examples/realsense/frame_agx_rig.yaml"
WARMUP_FRAMES = 100
FPS = 30
SYNC_MATCHING_THRESHOLD_MS = 20 * 1e6  # 5ms in nanoseconds
IMAGE_JITTER_THRESHOLD_MS = (1000 / FPS + 2) * 1e6  # nanoseconds


def load_camera_configuration() -> Dict:
    """Load camera configuration from YAML file.
    
    Returns:
        Dictionary containing camera configuration
    """
    print("Loading camera configuration...")
    with open(CONFIG_FILE, 'r') as file:
        return yaml.safe_load(file)


def setup_camera_pipelines(serial_numbers: List[str]) -> Tuple[
    List[rs.pipeline], List[rs.config]
]:
    """Set up pipelines for all cameras.
    
    Args:
        serial_numbers: List of camera serial numbers
        
    Returns:
        Tuple of (pipelines, configs)
    """
    print("Setting up camera pipelines...")
    pipelines = []
    configs = []
    
    for i, serial in enumerate(serial_numbers):
        pipeline, config = setup_pipeline(serial)
        pipelines.append(pipeline)
        configs.append(config)
        print(f"Camera {i+1} ({serial}) pipeline configured")
    
    return pipelines, configs


def extract_camera_parameters(
    pipelines: List[rs.pipeline],
    configs: List[rs.config],
    stereo_cameras: List[Dict]
) -> Dict[str, Dict]:
    """Extract intrinsics for all cameras.
    
    Args:
        pipelines: List of RealSense pipelines
        configs: List of RealSense configs
        stereo_cameras: List of stereo camera configurations
        
    Returns:
        Dictionary containing camera parameters
    """
    print("Extracting camera intrinsics...")
    camera_params = {}
    
    for i, (pipeline, config, stereo_cam) in enumerate(
        zip(pipelines, configs, stereo_cameras)
    ):
        left_intrinsics, right_intrinsics = get_camera_intrinsics(
            pipeline, config
        )
        camera_idx = f"camera_{i+1}"
        camera_params[camera_idx] = {
            "left": {
                "intrinsics": left_intrinsics,
                "extrinsics": stereo_cam["left_camera"]["transform"]
            },
            "right": {
                "intrinsics": right_intrinsics,
                "extrinsics": stereo_cam["right_camera"]["transform"]
            }
        }
        print(f"Camera {i+1} intrinsics extracted")
    
    return camera_params


def configure_all_devices(
    pipelines: List[rs.pipeline], configs: List[rs.config]
) -> None:
    """Configure all devices.
    
    Args:
        pipelines: List of RealSense pipelines
        configs: List of RealSense configs
    """
    print("Configuring devices...")
    for i, (pipeline, config) in enumerate(zip(pipelines, configs)):
        configure_device(pipeline, config, is_master=(i == 0))


def start_all_cameras(
    pipelines: List[rs.pipeline], configs: List[rs.config]
) -> None:
    """Start streaming from all cameras.
    
    Args:
        pipelines: List of RealSense pipelines
        configs: List of RealSense configs
    """
    print("Starting camera streams...")
    for pipeline, config in zip(pipelines, configs):
        pipeline.start(config)


def get_synchronized_frames(
    pipelines: List[rs.pipeline], frame_id: int
) -> Tuple[List[int], List[np.ndarray]]:
    """Get frames from all cameras.
    
    Args:
        pipelines: List of RealSense pipelines
        frame_id: Current frame ID
        
    Returns:
        Tuple of (timestamps, images)
    """
    all_timestamps = []
    all_images = []
    
    for i, pipeline in enumerate(pipelines):
        frames = pipeline.wait_for_frames()
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        
        if not left_frame or not right_frame:
            print(f"Skip frames from camera {i+1}, frame_id: {frame_id}")
            continue
            
        all_timestamps.append(int(left_frame.timestamp * 1e6))
        all_images.extend([
            np.asanyarray(left_frame.get_data()),
            np.asanyarray(right_frame.get_data())
        ])
    
    return all_timestamps, all_images


def check_timestamp_synchronization(timestamps: List[int]) -> None:
    """Check timestamp synchronization between cameras.
    
    Args:
        timestamps: List of timestamps from all cameras
    """
    max_timestamp_diff = 0
    for i in range(len(timestamps)):
        for j in range(i+1, len(timestamps)):
            timestamp_diff = abs(timestamps[i] - timestamps[j])
            max_timestamp_diff = max(max_timestamp_diff, timestamp_diff)
    
    if max_timestamp_diff > SYNC_MATCHING_THRESHOLD_MS:
        print(
            f"Warning: Timestamp synchronization failed, "
            f"max_timestamp_diff: {max_timestamp_diff/1e6:.2f} ms exceeds "
            f"threshold {SYNC_MATCHING_THRESHOLD_MS/1e6:.2f} ms"
        )


def main() -> None:
    """Main function for multi-camera tracking."""
    # Load configuration
    config_data = load_camera_configuration()
    stereo_cameras = config_data['stereo_cameras']
    serial_numbers = [cam['serial'] for cam in stereo_cameras]

    # Setup pipelines
    pipelines, configs = setup_camera_pipelines(serial_numbers)

    # Extract camera parameters
    camera_params = extract_camera_parameters(
        pipelines, configs, stereo_cameras
    )

    # Create rig with all cameras
    print("Creating camera rig...")
    rig = get_rs_multi_rig(camera_params)

    # Initialize tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=True
    )
    slam_cfg = vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)
    tracker = vslam.Tracker(rig, cfg, slam_cfg)

    # Configure all devices
    configure_all_devices(pipelines, configs)

    # Start all cameras
    start_all_cameras(pipelines, configs)

    # Initialize visualization and logging
    visualizer = RerunVisualizer(num_viz_cameras=len(serial_numbers))
    odom_logger = OdomLogger()
    slam_trajectory: List[np.ndarray] = []
    loop_closure_poses: List[np.ndarray] = []
    frame_id = 0
    prev_timestamp: Optional[int] = None

    try:
        print("Start running")
        while True:
            frame_id += 1
            
            # Get frames from all cameras
            all_timestamps, all_images = get_synchronized_frames(
                pipelines, frame_id
            )
            
            # Warmup period for camera synchronization
            if frame_id < WARMUP_FRAMES:
                continue

            # Skip if any camera frames are missing
            if len(all_timestamps) < len(pipelines):
                continue

            # Check for frame drops
            if prev_timestamp is not None:
                timestamp_diff = all_timestamps[0] - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_MS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_MS/1e6:.2f} ms"
                    )

            # Check timestamp synchronization
            check_timestamp_synchronization(all_timestamps)
            
            # Track frame using the first timestamp
            vo_pose_estimate, slam_pose = tracker.track(all_timestamps[0], tuple(all_images))

            if vo_pose_estimate.world_from_rig is None:
                continue

            if slam_pose is None:
                continue

            slam_trajectory.append(slam_pose.translation)

            odom_logger.log(frame_id, Pose(vo_pose_estimate.world_from_rig.pose), Pose(slam_pose))

            current_lc_poses = tracker.get_loop_closure_poses()
            if current_lc_poses and (
                not loop_closure_poses
                or not np.array_equal(
                    current_lc_poses[-1].pose.translation, loop_closure_poses[-1]
                )
            ):
                loop_closure_poses.append(current_lc_poses[-1].pose.translation)

            # Visualize results (showing only left cameras)
            left_images = [all_images[i] for i in range(0, len(all_images), 2)]
            left_observations = [
                tracker.get_last_observations(i)
                for i in range(0, len(all_images), 2)
            ]

            visualizer.visualize_frame(
                frame_id=frame_id,
                images=left_images,
                observations_main_cam=left_observations,
                slam_pose=slam_pose,
                slam_trajectory=slam_trajectory,
                timestamp=all_timestamps[0],
                final_landmarks=tracker.get_final_landmarks(),
                loop_closure_poses=(
                    loop_closure_poses if loop_closure_poses else None
                ),
            )
            
            # Store current timestamp for next iteration
            prev_timestamp = deepcopy(all_timestamps[0])

    finally:
        odom_logger.close()
        for pipeline in pipelines:
            pipeline.stop()


if __name__ == "__main__":
    main()
