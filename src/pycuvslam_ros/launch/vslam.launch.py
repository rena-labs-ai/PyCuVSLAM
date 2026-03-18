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
        description="Path to rig extrinsics YAML (stereo_cameras with left_camera.transform, right_camera.transform)",
    )
    camera_topics_arg = DeclareLaunchArgument(
        "camera_topics",
        default_value="/front/camera/infra1/image_rect_raw/compressed /front/camera/infra2/image_rect_raw/compressed /left/camera/infra1/image_rect_raw/compressed /left/camera/infra2/image_rect_raw/compressed /right/camera/infra1/image_rect_raw/compressed /right/camera/infra2/image_rect_raw/compressed /back/camera/infra1/image_rect_raw/compressed /back/camera/infra2/image_rect_raw/compressed",
        description="Space-separated compressed topics: left1 right1 [left2 right2 ...]",
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

    vslam_node = Node(
        package="pycuvslam_ros",
        executable="vslam_node",
        name="vslam",
        output="screen",
        parameters=[
            {
                "config_file": LaunchConfiguration("config_file"),
                "camera_topics": LaunchConfiguration("camera_topics"),
                "enable_visualization": LaunchConfiguration("enable_visualization"),
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
        camera_topics_arg,
        experiment_arg,
        log_dir_arg,
        enable_viz_arg,
        vslam_node,
        odom_diff_logger,
    ])
