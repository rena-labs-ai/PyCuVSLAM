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

The Ros Multicam Tracker node needs the camera topic being published. Hence:

```bash
ros2 launch pycuvslam_ros camera.launch.py serials:="<front-camera-serial>,<left-camera-serial>,<right-camera-serial>,<back-camera-serial>"
```

If having less cameras (e.g., only front and left), only pass the corresponding serial numbers.


Update the camera [rig confg](./src/pycuvslam_ros/config/frame_agx_rig.yaml) (e.g., if having only front and left, comment back and right cameras).

Finally run the vslam multicam tracker:

```bash
ros2 launch pycuvslam_ros vslam.launch.py
```

If having <4 cameras (e.g., only front and left):

```bash
ros2 launch pycuvslam_ros vslam.launch.py camera_topics:="/front/camera/infra1/image_rect_raw/compressed /front/camera/infra2/image_rect_raw/compressed /left/camera/infra1/image_rect_raw/compressed /left/camera/infra2/image_rect_raw/compressed"
```

It creates a live image comparing cuvslam/odometry with fast_lio /Odometry (make sure fast_lio is running).
