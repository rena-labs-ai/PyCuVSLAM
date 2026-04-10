"""Launch file for vslam. Assumes cameras are already running."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _topics_for_camera(camera: str) -> dict[str, str]:
    zed_stem = (
        f"/{camera}_base/zed_node"
        if camera in ("zedm", "zed2i")
        else "/zed/zed_node"
    )
    return {
        "zed_left_topic": f"{zed_stem}/left/color/rect/image/compressed",
        "zed_right_topic": f"{zed_stem}/right/color/rect/image/compressed",
        "zed_imu_topic": f"{zed_stem}/imu/data",
        "hawk_left_topic": "/left/image_rect",
        "hawk_right_topic": "/right/image_rect",
    }


def generate_launch_description():
    pkg_share = get_package_share_directory("pycuvslam_ros")
    default_config = os.path.join(pkg_share, "config", "frame_agx_rig.yaml")
    default_hawk_rig = os.path.join(pkg_share, "config", "hawk_rig.yaml")

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to rig YAML (stereo_cameras with name, serial, left/right_camera.transform). Topics derived as /{name}/camera/infra1|2/image_rect_raw/compressed",
    )
    experiment_arg = DeclareLaunchArgument(
        "experiment",
        default_value="default",
        description="Experiment name for odom diff logging",
    )
    log_dir_arg = DeclareLaunchArgument(
        "log_dir",
        default_value="./outputs",
        description="Directory for odom diff logging",
    )
    enable_viz_arg = DeclareLaunchArgument(
        "enable_visualization",
        default_value="false",
        description="Enable Rerun visualization",
    )
    tracker_arg = DeclareLaunchArgument(
        "tracker",
        default_value="ros_multicam",
        description="Tracker: ros_multicam, ros_zed_stereo, ros_zed_vio, ros_hawk_stereo, or ros_hawk_multicam",
    )
    hawk_rig_arg = DeclareLaunchArgument(
        "hawk_rig_file",
        default_value=default_hawk_rig,
        description="Path to hawk rig YAML (for ros_hawk_multicam). Topics and extrinsics per stereo pair.",
    )
    camera_arg = DeclareLaunchArgument(
        "camera",
        default_value="zed2i",
        description="Camera preset: zedm, zed2i (ZED topics under /{camera}_base/zed_node/...), or hawk (HAWK stereo on /left|/right/image_rect).",
        choices=["zedm", "zed2i", "hawk"],
    )
    base_link_arg = DeclareLaunchArgument(
        "base_link_frame",
        default_value="zed_camera_link",
        description="Child frame for odom TF (odom -> base_link_frame)",
    )

    def launch_vslam(context, *args, **kwargs):
        camera = LaunchConfiguration("camera").perform(context)
        topics = _topics_for_camera(camera)
        return [
            Node(
                package="pycuvslam_ros",
                executable="vslam_node",
                name="vslam",
                output="screen",
                parameters=[
                    {
                        "config_file": LaunchConfiguration("config_file"),
                        "enable_visualization": LaunchConfiguration(
                            "enable_visualization"
                        ),
                        "tracker": LaunchConfiguration("tracker"),
                        "hawk_rig_file": LaunchConfiguration("hawk_rig_file"),
                        "base_link_frame": LaunchConfiguration("base_link_frame"),
                        **topics,
                    }
                ],
            )
        ]

    odom_diff_logger = Node(
        package="pycuvslam_ros",
        executable="odom_diff_logger",
        name="odom_diff_logger",
        parameters=[
            {
                "experiment": LaunchConfiguration("experiment"),
                "log_dir": LaunchConfiguration("log_dir"),
                "ref_odom_topic": "/Odometry",
                "other_odom_topic": "/cuvslam/odometry",
            }
        ],
    )

    return LaunchDescription([
        config_file_arg,
        experiment_arg,
        log_dir_arg,
        enable_viz_arg,
        tracker_arg,
        hawk_rig_arg,
        camera_arg,
        base_link_arg,
        OpaqueFunction(function=launch_vslam),
        odom_diff_logger,
    ])
