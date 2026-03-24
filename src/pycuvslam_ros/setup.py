from setuptools import setup, find_packages

package_name = "pycuvslam_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["setuptools", "matplotlib"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            ["launch/vslam.launch.py", "launch/camera.launch.py", "launch/zed_camera.launch.py"],
        ),
        (
            "share/" + package_name + "/config",
            ["config/frame_agx_rig.yaml", "config/zed_common.yaml", "config/zed2.yaml"],
        ),
    ],
    zip_safe=True,
    maintainer="User",
    maintainer_email="user@example.com",
    description="ROS2 wrapper for PyCuVSLAM multi-camera visual SLAM",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vslam_node = pycuvslam_ros.vslam_node:main",
            "odom_diff_logger = pycuvslam_ros.odom_logger:main",
            "camera_hz_logger = pycuvslam_ros.camera_hz_logger:main",
        ],
    },
)
