# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: no monitor required
import matplotlib.pyplot as plt
import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

import cuvslam as vslam

# Constants
FPS = 30
RESOLUTION = (1280, 720)
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_NS = 35 * 1e6  # 35ms in nanoseconds

# Camera border margins to exclude features near image edges
BORDER_TOP = 50
BORDER_BOTTOM = 0
BORDER_LEFT = 70
BORDER_RIGHT = 70

# Conversion factor from cm to meters
CM_TO_METERS = 100

# Depth scale: OAK-D StereoDepth outputs depth in millimeters by default
DEPTH_SCALE_FACTOR = 1000

# Output trajectory plot
TRAJECTORY_PLOT_PATH = "trajectory.png"


def oak_transform_to_pose(oak_extrinsics: List[List[float]]) -> vslam.Pose:
    """Convert 4x4 transformation matrix to cuVSLAM pose."""
    extrinsics_array = np.array(oak_extrinsics)
    rotation_matrix = extrinsics_array[:3, :3]
    translation_vector = extrinsics_array[:3, 3] / CM_TO_METERS  # to meters

    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_vector)


def set_cuvslam_camera(oak_params: Dict[str, Any]) -> vslam.Camera:
    """Create a Camera object from OAK camera parameters."""
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(
        vslam.Distortion.Model.Polynomial, oak_params['distortion']
    )
    cam.focal = (
        oak_params['intrinsics'][0][0],
        oak_params['intrinsics'][1][1]
    )
    cam.principal = (
        oak_params['intrinsics'][0][2],
        oak_params['intrinsics'][1][2]
    )
    cam.size = oak_params['resolution']
    cam.rig_from_camera = oak_transform_to_pose(oak_params['extrinsics'])

    cam.border_top = BORDER_TOP
    cam.border_bottom = BORDER_BOTTOM
    cam.border_left = BORDER_LEFT
    cam.border_right = BORDER_RIGHT
    return cam


def get_rgb_calibration(
    calib_data: dai.CalibrationHandler, resolution: Tuple[int, int]
) -> Dict[str, Dict[str, Any]]:
    """Get RGB calibration. Color (CAM_A) is the rig origin."""
    rgb_camera = {'color': {}}
    rgb_camera['color']['resolution'] = resolution
    rgb_camera['color']['intrinsics'] = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, resolution[0], resolution[1]
    )
    rgb_camera['color']['extrinsics'] = np.eye(4).tolist()
    rgb_camera['color']['distortion'] = calib_data.getDistortionCoefficients(
        dai.CameraBoardSocket.CAM_A
    )[:8]
    return rgb_camera


def save_trajectory_plot(trajectory: List[np.ndarray], path: str) -> None:
    """Save a top-down (X-Y) trajectory plot to disk."""
    if not trajectory:
        print("No trajectory to plot.")
        return

    traj = np.asarray(trajectory)
    xs, ys = traj[:, 2], -traj[:, 0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, '-', linewidth=1.0, label="trajectory")
    ax.plot(xs[0], ys[0], 'go', markersize=8, label="start")
    ax.plot(xs[-1], ys[-1], 'r*', markersize=12, label="end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"OAK-D RGBD trajectory ({len(trajectory)} poses)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot to {path}")


def main() -> None:
    """Main function for OAK-D RGBD tracking."""
    device = dai.Device()
    device.setIrLaserDotProjectorIntensity(0.8)   # 0.0 to 1.0
    device.setIrFloodLightIntensity(0.0)          # 0.0 to 1.0
    calib_data = device.readCalibration()
    rgb_camera = get_rgb_calibration(calib_data, RESOLUTION)

    cameras = [set_cuvslam_camera(rgb_camera['color'])]

    rgbd_settings = vslam.Tracker.OdometryRGBDSettings()
    rgbd_settings.depth_scale_factor = DEPTH_SCALE_FACTOR
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = False

    cfg = vslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        odometry_mode=vslam.Tracker.OdometryMode.RGBD,
        rgbd_settings=rgbd_settings
    )
    tracker = vslam.Tracker(vslam.Rig(cameras), cfg)

    pipeline = dai.Pipeline(device)

    color_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A, sensorFps=FPS
    )
    left_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorFps=FPS
    )
    right_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C, sensorFps=FPS
    )

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left_camera.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8),
        right_camera.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8),
        presetMode=dai.node.StereoDepth.PresetMode.FAST_DENSITY,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)

    color_output = color_camera.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.BGR888i)

    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.005))
    color_output.link(sync.inputs["color"])
    stereo.depth.link(sync.inputs["depth"])

    sync_queue = sync.out.createOutputQueue()

    frame_id = 0
    prev_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []

    pipeline.start()
    print("Tracking... Ctrl+C to stop and save trajectory plot.")

    try:
        while pipeline.isRunning():
            message_group: dai.MessageGroup = sync_queue.get()
            color_frame = message_group["color"]
            depth_frame = message_group["depth"]

            timestamp_ns = int(message_group.getTimestamp().total_seconds() * 1e9)

            if prev_timestamp is not None:
                timestamp_diff = timestamp_ns - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms"
                    )

            frame_id += 1

            if frame_id > WARMUP_FRAMES:
                color_img = color_frame.getCvFrame()
                depth_img = depth_frame.getFrame()

                odom_pose_estimate, _ = tracker.track(
                    timestamp_ns, images=[color_img], depths=[depth_img]
                )

                odom_pose_with_cov = odom_pose_estimate.world_from_rig
                if odom_pose_with_cov is None:
                    print(f"Tracking failed at frame {frame_id}")
                    continue

                t = odom_pose_with_cov.pose.translation
                trajectory.append(np.asarray(t))

            prev_timestamp = timestamp_ns
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        save_trajectory_plot(trajectory, TRAJECTORY_PLOT_PATH)


if __name__ == "__main__":
    main()