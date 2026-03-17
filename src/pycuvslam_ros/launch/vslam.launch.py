"""Launch file for vslam with RosMulticamTracker. Assumes cameras are already running."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value="cuvslam_examples/cuvslam_examples/realsense/frame_agx_rig.yaml",
        description="Path to rig extrinsics YAML (stereo_cameras with left_camera.transform, right_camera.transform)",
    )
    camera_topics_arg = DeclareLaunchArgument(
        "camera_topics",
        default_value="/front/camera/infra1/image_rect_raw /front/camera/infra2/image_rect_raw /left/camera/infra1/image_rect_raw /left/camera/infra2/image_rect_raw /right/camera/infra1/image_rect_raw /right/camera/infra2/image_rect_raw /back/camera/infra1/image_rect_raw /back/camera/infra2/image_rect_raw",
        description="Space-separated topics: left1 right1 [left2 right2 ...]",
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

    vslam_node = Node(
        package="pycuvslam_ros",
        executable="vslam_node",
        name="vslam",
        output="screen",
        parameters=[
            {
                "config_file": LaunchConfiguration("config_file"),
                "camera_topics": LaunchConfiguration("camera_topics"),
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
        vslam_node,
        odom_diff_logger,
    ])
