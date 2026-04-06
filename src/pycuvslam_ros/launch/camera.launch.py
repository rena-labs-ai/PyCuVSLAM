"""Launch RealSense cameras for MultiCameraTracker/RosMulticamTracker.

Reads rig config (stereo_cameras with serial + name), configures IR-only stereo
(infra1, infra2), 640x480@30fps, manual exposure 10ms. Topics: /{name}/camera/infra1|2/image_rect_raw.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# MultiCameraTracker requirements from tracker.py / camera_utils.py
RESOLUTION = "640x480"
FPS = 30
IR_EXPOSURE_US = 10000
# realsense2_camera: 1=Master, 2=Slave, 0=Default (no sync)
SYNC_MODE_MASTER = "1"
SYNC_MODE_SLAVE = "2"


def _load_cameras_from_config(config_path: str) -> list[tuple[str, str]]:
    """Returns list of (serial, namespace) from stereo_cameras. Namespace = name or serial."""
    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f)
    result = []
    for c in data["stereo_cameras"]:
        serial = str(c["serial"])
        namespace = str(c.get("name", serial)).strip("/")
        result.append((serial, namespace))
    return result


def _launch_cameras(context, *args, **kwargs):
    config_file = LaunchConfiguration("config_file").perform(context)
    if not config_file:
        raise ValueError("config_file is required")

    cameras = _load_cameras_from_config(config_file)
    if not cameras:
        raise ValueError("config_file has no stereo_cameras")

    nodes = []
    for i, (serial, namespace) in enumerate(cameras):
        is_master = i == 0
        params = {
            "serial_no": serial,
            "enable_color": False,
            "enable_depth": False,
            "enable_infra1": True,
            "enable_infra2": True,
            "depth_module.depth_profile": f"640,480,{FPS}",
            "depth_module.infra_profile": f"640,480,{FPS}",
            "depth_module.enable_auto_exposure": True,
            "depth_module.emitter_enabled": 0,
            "enable_gyro": False,
            "enable_accel": False,
            "enable_sync": True,
            "depth_module.inter_cam_sync_mode": (
                SYNC_MODE_MASTER if is_master else SYNC_MODE_SLAVE
            ),
            "publish_tf": False,
            # "depth_module.exposure": IR_EXPOSURE_US,
            # "infra_qos": "SENSOR_DATA",  # best-effort for infra1/infra2
        }
        node = Node(
            package="realsense2_camera",
            namespace=namespace,
            name="camera",
            executable="realsense2_camera_node",
            parameters=[params],
            output="screen",
        )
        nodes.append(node)

    return [GroupAction(nodes)]


def generate_launch_description():
    pkg_share = get_package_share_directory("pycuvslam_ros")
    default_config = os.path.join(pkg_share, "config", "frame_agx_rig.yaml")

    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Rig YAML (stereo_cameras with serial + name). Topics: /{name}/camera/infra1|2/image_rect_raw",
    )

    return LaunchDescription([
        config_arg,
        OpaqueFunction(function=_launch_cameras),
    ])
