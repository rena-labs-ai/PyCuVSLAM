"""Launch ZED stereo camera for PyCuVSLAM (ZED 2 only).

Loads ``pycuvslam_ros`` ``config/zed_common.yaml`` and a model YAML under ``config/``
(see ``camera_config`` argument, e.g. ``zed2.yaml`` or ``zed2_low_light.yaml``), plus
``zed_wrapper`` object-detection YAMLs, matching the nvblox-style parameter list.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ComposableNode

ZED_CAMERA_NAME = "zed"
ZED_CAMERA_MODEL = "zed2"
CONTAINER_NAME = "zed_container"


def generate_launch_description() -> LaunchDescription:
    pyc_share = get_package_share_directory("pycuvslam_ros")
    zed_wrapper_share = get_package_share_directory("zed_wrapper")
    zed_desc_share = get_package_share_directory("zed_description")

    camera_config_arg = DeclareLaunchArgument(
        "camera_config",
        default_value="zed2.yaml",
        description="Filename under pycuvslam_ros/config (e.g. zed2.yaml, zed2_low_light.yaml).",
    )
    config_common = os.path.join(pyc_share, "config", "zed_common.yaml")
    config_camera = PathJoinSubstitution(
        [FindPackageShare("pycuvslam_ros"), "config", LaunchConfiguration("camera_config")]
    )
    obj_det = os.path.join(zed_wrapper_share, "config", "object_detection.yaml")
    custom_obj = os.path.join(zed_wrapper_share, "config", "custom_object_detection.yaml")

    xacro_path = os.path.join(zed_desc_share, "urdf", "zed_descr.urdf.xacro")

    rsp = Node(
        package="robot_state_publisher",
        namespace=ZED_CAMERA_NAME,
        executable="robot_state_publisher",
        name="zed_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": Command(
                    [
                        "xacro",
                        " ",
                        xacro_path,
                        " ",
                        "camera_name:=",
                        ZED_CAMERA_NAME,
                        " ",
                        "camera_model:=",
                        ZED_CAMERA_MODEL,
                    ]
                )
            }
        ],
        remappings=[("robot_description", f"{ZED_CAMERA_NAME}_description")],
    )

    zed_node = ComposableNode(
        package="zed_components",
        namespace=ZED_CAMERA_NAME,
        name="zed_node",
        plugin="stereolabs::ZedCamera",
        parameters=[
            config_common,
            config_camera,
            obj_det,
            custom_obj,
            {
                "general.camera_name": ZED_CAMERA_NAME,
                "general.camera_model": ZED_CAMERA_MODEL,
            },
        ],
    )

    zed_container = ComposableNodeContainer(
        name=CONTAINER_NAME,
        namespace=ZED_CAMERA_NAME,
        package="rclcpp_components",
        executable="component_container_isolated",
        arguments=["--use_multi_threaded_executor", "--ros-args", "--log-level", "info"],
        output="screen",
        composable_node_descriptions=[],
    )

    load_zed = LoadComposableNodes(
        target_container=f"/{ZED_CAMERA_NAME}/{CONTAINER_NAME}",
        composable_node_descriptions=[zed_node],
    )

    return LaunchDescription([camera_config_arg, rsp, zed_container, load_zed])
