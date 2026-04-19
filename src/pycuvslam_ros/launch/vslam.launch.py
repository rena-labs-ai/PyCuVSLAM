"""Launch file for vslam. Assumes cameras are already running."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("pycuvslam_ros")
    default_rig_base = os.path.join(pkg_share, "config")

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=os.path.join(pkg_share, "config", "frame_agx_rig.yaml"),
        description="Path to rig YAML (ros_multicam only).",
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
        description="Tracker id.",
    )
    rig_base_path_arg = DeclareLaunchArgument(
        "rig_base_path",
        default_value=default_rig_base,
        description="Directory containing per-tracker rig YAMLs (hawk_rig.yaml, "
                    "oak_rig.yaml, etc.); the tracker picks its own file from here.",
    )
    camera_arg = DeclareLaunchArgument(
        "camera",
        default_value="zed2i",
        description="Camera model id.",
        choices=["zed", "zedm", "zed2i", "hawk", "oak",
                 "realsensed435", "realsensed455"],
    )
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="/cuvslam/odometry",
        description="Topic to publish odometry on (remapped from /cuvslam/odometry).",
    )
    enable_plot_arg = DeclareLaunchArgument(
        "enable_plot",
        default_value="false",
        description="Launch odom_diff_logger to plot tracker vs ref.",
    )
    ref_odom_topic_arg = DeclareLaunchArgument(
        "ref_odom_topic",
        default_value="/Odometry",
        description="Reference odometry topic for the plot.",
    )
    plot_out_path_arg = DeclareLaunchArgument(
        "plot_out_path",
        default_value="./outputs/vslam_plot.png",
        description="Output PNG path for the tracker-vs-ref plot.",
    )

    def launch_vslam(context, *args, **kwargs):
        odom_topic = context.launch_configurations.get("odom_topic", "/cuvslam/odometry")
        tracker_name = context.launch_configurations.get("tracker", "cuvslam")
        camera_name = context.launch_configurations.get("camera", "zed2i")
        camera_link = f"{camera_name}_camera_link"
        enable_plot = context.launch_configurations.get("enable_plot", "false").lower() == "true"

        actions = [
            Node(
                package="pycuvslam_ros",
                executable="vslam_node",
                name="vslam",
                output="screen",
                parameters=[
                    {
                        "config_file": LaunchConfiguration("config_file"),
                        "enable_visualization": LaunchConfiguration("enable_visualization"),
                        "tracker": LaunchConfiguration("tracker"),
                        "camera": LaunchConfiguration("camera"),
                        "rig_base_path": LaunchConfiguration("rig_base_path"),
                        "camera_link": camera_link,
                    }
                ],
                remappings=[("/cuvslam/odometry", odom_topic)],
            )
        ]

        if enable_plot:
            actions.append(
                Node(
                    package="pycuvslam_ros",
                    executable="odom_diff_logger",
                    name="odom_diff_logger",
                    output="screen",
                    parameters=[{
                        "ref_odom_topic": LaunchConfiguration("ref_odom_topic"),
                        "estimated_odom_topics": odom_topic,
                        "estimated_labels": tracker_name,
                        "out_path": LaunchConfiguration("plot_out_path"),
                        "title": LaunchConfiguration("experiment"),
                    }],
                )
            )
        return actions

    return LaunchDescription(
        [
            config_file_arg,
            experiment_arg,
            log_dir_arg,
            enable_viz_arg,
            tracker_arg,
            rig_base_path_arg,
            camera_arg,
            odom_topic_arg,
            enable_plot_arg,
            ref_odom_topic_arg,
            plot_out_path_arg,
            OpaqueFunction(function=launch_vslam),
        ]
    )
