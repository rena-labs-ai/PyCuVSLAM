"""Launch file for vslam. Assumes the OAK camera is already running."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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
        default_value="ros_oak_rgbd",
        description="OAK tracker id.",
        choices=["ros_oak_stereo", "ros_oak_rgbd"],
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
    odom_child_frame_arg = DeclareLaunchArgument(
        "odom_child_frame",
        default_value="nav_base_link",
        description="Odometry child_frame_id; match cuvslam_sync_nav_base_link (default nav_base_link).",
    )

    def launch_vslam(context, *args, **kwargs):
        odom_topic = context.launch_configurations.get("odom_topic", "/cuvslam/odometry")
        tracker_name = context.launch_configurations.get("tracker", "ros_oak_rgbd")
        enable_plot = context.launch_configurations.get("enable_plot", "false").lower() == "true"

        actions = [
            Node(
                package="pycuvslam_ros",
                executable="vslam_node",
                name="vslam",
                output="screen",
                parameters=[
                    {
                        "enable_visualization": LaunchConfiguration("enable_visualization"),
                        "tracker": LaunchConfiguration("tracker"),
                        "odom_child_frame": LaunchConfiguration("odom_child_frame"),
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
            experiment_arg,
            log_dir_arg,
            enable_viz_arg,
            tracker_arg,
            odom_topic_arg,
            enable_plot_arg,
            ref_odom_topic_arg,
            plot_out_path_arg,
            odom_child_frame_arg,
            OpaqueFunction(function=launch_vslam),
        ]
    )
