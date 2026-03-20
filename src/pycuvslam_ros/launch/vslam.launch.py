"""Launch file for vslam with RosMulticamTracker. Assumes cameras are already running."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("pycuvslam_ros")
    default_config = os.path.join(pkg_share, "config", "frame_agx_rig.yaml")

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
        default_value="./",
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
        description="Tracker: ros_multicam or ros_zed_stereo",
    )
    zed_left_arg = DeclareLaunchArgument(
        "zed_left_topic",
        default_value="/zed/zed_node/left/color/rect/image/compressed",
        description="ZED left image topic (for ros_zed_stereo)",
    )
    zed_right_arg = DeclareLaunchArgument(
        "zed_right_topic",
        default_value="/zed/zed_node/right/color/rect/image/compressed",
        description="ZED right image topic (for ros_zed_stereo)",
    )
    base_link_arg = DeclareLaunchArgument(
        "base_link_frame",
        default_value="zed_camera_link",
        description="Child frame for odom TF (odom -> base_link_frame)",
    )
    vslam_node = Node(
        package="pycuvslam_ros",
        executable="vslam_node",
        name="vslam",
        output="screen",
        parameters=[
            {
                "config_file": LaunchConfiguration("config_file"),
                "enable_visualization": LaunchConfiguration("enable_visualization"),
                "tracker": LaunchConfiguration("tracker"),
                "zed_left_topic": LaunchConfiguration("zed_left_topic"),
                "zed_right_topic": LaunchConfiguration("zed_right_topic"),
                "base_link_frame": LaunchConfiguration("base_link_frame"),
            }
        ],
    )

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
        zed_left_arg,
        zed_right_arg,
        base_link_arg,
        vslam_node,
        odom_diff_logger,
    ])
