"""Launch RealSense cameras for MultiCameraTracker/RosMulticamTracker.

Configures IR-only stereo (infra1, infra2), 640x480@30fps, manual exposure 10ms,
first camera as sync master, others as slaves. Topics: /serial_XXX/camera/infra1|2/image_rect_raw.
"""

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


def _load_serials_from_config(config_path: str) -> list[str]:
    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return [str(c["serial"]) for c in data["stereo_cameras"]]


def _launch_cameras(context, *args, **kwargs):
    config_file = LaunchConfiguration("config_file").perform(context)
    serial_numbers = LaunchConfiguration("serial_numbers").perform(context)

    if config_file:
        serials = _load_serials_from_config(config_file)
    else:
        serials = [s.strip() for s in serial_numbers.split(",") if s.strip()]

    if not serials:
        raise ValueError("Provide config_file or serial_numbers (space-separated)")

    nodes = []
    for i, serial in enumerate(serials):
        is_master = i == 0
        namespace = f"serial_{serial}"
        params = {
            "serial_no": serial,
            "enable_color": "false",
            "enable_depth": "false",
            "enable_infra1": "true",
            "enable_infra2": "true",
            "depth_module.depth_profile": f"640,480,{FPS}",
            "depth_module.infra_profile": f"640,480,{FPS}",
            "depth_module.exposure": str(IR_EXPOSURE_US),
            "depth_module.enable_auto_exposure": "false",
            "enable_sync": "true",
            "depth_module.inter_cam_sync_mode": SYNC_MODE_MASTER if is_master else SYNC_MODE_SLAVE,
            "publish_tf": "false",
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
    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value="",
        description="Rig YAML (frame_agx_rig.yaml) to extract serial numbers; overrides serial_numbers",
    )
    serials_arg = DeclareLaunchArgument(
        "serial_numbers",
        default_value="242422303248,146222250568,339522301389",
        description="Comma-separated serial numbers (used if config_file empty)",
    )

    return LaunchDescription([
        config_arg,
        serials_arg,
        OpaqueFunction(function=_launch_cameras),
    ])
