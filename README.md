# Rena Pycuvslam ROS Node

## Build

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

build cuvslam ros package:

```bash
colcon build --packages-select pycuvslam_ros
```

## Run

Only OAK cameras are supported. The camera topics are derived inside the
tracker from `rena_bringup/config/config.yaml` (serial + `image_mode` →
`image_raw` | `image_rect`), so there is no separate rig config to edit.

1. Bring up the OAK camera (via the robot bring-up, e.g. `rena start`, which
   launches the `depthai_ros_driver_v3` OAK driver).

2. Launch vslam:

```bash
# RGBD tracker (default)
ros2 launch pycuvslam_ros vslam.launch.py

# or the stereo tracker
ros2 launch pycuvslam_ros vslam.launch.py tracker:=ros_oak_stereo
```

With `enable_plot:=true` it creates a live image comparing `/cuvslam/odometry`
with fast_lio `/Odometry` (make sure fast_lio is running).
