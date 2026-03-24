"""Launch ZED stereo camera for PyCuVSLAM.

Wraps Stereolabs ``zed_wrapper/launch/zed_camera.launch.py`` with defaults aligned to the
nvblox ZED example (namespace ``zed``, node ``zed_node``). Does not depend on nvblox or
isaac_ros_launch_utils (Humble vs Jazzy).

Camera model names mirror ``NvbloxCamera.zed2`` / ``NvbloxCamera.zedx`` in
``nvblox_ros_python_utils.nvblox_launch_utils`` without importing that package.
"""

from __future__ import annotations

import os
from enum import IntEnum

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


class ZedCameraModel(IntEnum):
    """ZED variants for launch args (ordinal 3/4 matches nvblox ``NvbloxCamera``)."""

    zed2 = 3
    zedx = 4

    @property
    def model_id(self) -> str:
        return self.name  # "zed2" or "zedx"


def generate_launch_description() -> LaunchDescription:
    zed_share = get_package_share_directory("zed_wrapper")
    zed_launch = os.path.join(zed_share, "launch", "zed_camera.launch.py")

    camera_model_arg = DeclareLaunchArgument(
        "camera_model",
        default_value=ZedCameraModel.zed2.model_id,
        description="ZED model: zed2 or zedx (same strings as zed_wrapper / nvblox).",
        choices=[ZedCameraModel.zed2.model_id, ZedCameraModel.zedx.model_id],
    )

    included = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_launch),
        launch_arguments={
            "camera_name": "zed",
            "camera_model": LaunchConfiguration("camera_model"),
            "node_name": "zed_node",
        },
    )

    return LaunchDescription([camera_model_arg, included])
