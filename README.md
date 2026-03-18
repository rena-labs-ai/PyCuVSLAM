# Rena Pycuvslam ROS Node

## Setup

First install micromamba:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

install ros2 (jazzy) and activate environment

```bash
# Create a ros-jazzy desktop environment
micromamba create -n ros_env -c conda-forge -c robostack-jazzy ros-jazzy-desktop python=3.12
# Activate the environment
micromamba activate ros_env
# Add the robostack channel to the environemnt
micromamba config append channels robostack-jazzy --env
```

install pycuvslam, make sure you have cu13 installed (given x86 or arm)

```bash
# aarch
pip install https://github.com/nvidia-isaac/cuVSLAM/releases/download/v15.0.0/cuvslam-15.0.0+cu13-cp312-abi3-manylinux_2_39_aarch64.whl
# x86
pip install https://github.com/nvidia-isaac/cuVSLAM/releases/download/v15.0.0/cuvslam-15.0.0+cu13-cp312-abi3-manylinux_2_39_x86_64.whl
```

install the cuvslam examples and its requirements:

```bash
pip install -r cuvslam_examples/requirements.txt
pip install -e cuvslam_examples
```

## Usage

1. Edit the [rig config](./src/pycuvslam_ros/config/frame_agx_rig.yaml): set `serial` and `name` per camera; comment out cameras you don't have.

2. Launch cameras:

```bash
ros2 launch pycuvslam_ros camera.launch.py
```

3. Launch vslam:

```bash
ros2 launch pycuvslam_ros vslam.launch.py
```

It creates a live image comparing cuvslam/odometry with fast_lio /Odometry (make sure fast_lio is running).
