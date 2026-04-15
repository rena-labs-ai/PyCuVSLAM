"""Launch ArgusStereoNode + left/right RectifyNode in a single container.

Publishes (among others):
  /left/image_raw        sensor_msgs/Image   (raw from argus)
  /left/camera_info      sensor_msgs/CameraInfo
  /left/image_rect       sensor_msgs/Image   (rectified)
  /left/camera_info_rect sensor_msgs/CameraInfo (rectified)
  (same for /right/*)

Pass calibration YAML files via left_camera_info_url / right_camera_info_url
to get accurate rectification; omit for nominal factory calibration.
"""

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'module_id',
            default_value='-1',
            description='Index specifying the stereo camera module to use.',
        ),
        DeclareLaunchArgument(
            'camera_id',
            default_value='0',
            description='Camera ID within the module.',
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='0',
            description='Argus sensor mode index (e.g. resolution / FPS preset).',
        ),
        DeclareLaunchArgument(
            'fsync_type',
            default_value='1',
            description='Frame sync type: 0 for internal, 1 for external.',
        ),
        DeclareLaunchArgument(
            'use_hw_timestamp',
            default_value='false',
            description='Use hardware timestamps when available.',
        ),
        DeclareLaunchArgument(
            'camera_link_frame_name',
            default_value='stereo_camera',
            description='TF frame name for the stereo camera link.',
        ),
        DeclareLaunchArgument(
            'left_camera_frame_name',
            default_value='stereo_camera_left',
            description='TF frame name for the left optical frame.',
        ),
        DeclareLaunchArgument(
            'right_camera_frame_name',
            default_value='stereo_camera_right',
            description='TF frame name for the right optical frame.',
        ),
        DeclareLaunchArgument(
            'left_camera_info_url',
            default_value='',
            description='URL for the left camera calibration YAML.',
        ),
        DeclareLaunchArgument(
            'right_camera_info_url',
            default_value='',
            description='URL for the right camera calibration YAML.',
        ),
        DeclareLaunchArgument(
            'wide_fov',
            default_value='false',
            description='Wide FoV mode when supported by the sensor.',
        ),
        DeclareLaunchArgument(
            'output_width',
            default_value='1920',
            description='Rectified output width (px).',
        ),
        DeclareLaunchArgument(
            'output_height',
            default_value='1200',
            description='Rectified output height (px).',
        ),
        DeclareLaunchArgument(
            'camera_namespace',
            default_value='front',
            description='Top-level namespace for all camera topics (e.g. front, rear).',
        ),
    ]

    module_id = LaunchConfiguration('module_id')
    camera_id = LaunchConfiguration('camera_id')
    mode = LaunchConfiguration('mode')
    fsync_type = LaunchConfiguration('fsync_type')
    use_hw_timestamp = LaunchConfiguration('use_hw_timestamp')
    camera_link_frame_name = LaunchConfiguration('camera_link_frame_name')
    left_camera_frame_name = LaunchConfiguration('left_camera_frame_name')
    right_camera_frame_name = LaunchConfiguration('right_camera_frame_name')
    left_camera_info_url = LaunchConfiguration('left_camera_info_url')
    right_camera_info_url = LaunchConfiguration('right_camera_info_url')
    wide_fov = LaunchConfiguration('wide_fov')
    output_width = LaunchConfiguration('output_width')
    output_height = LaunchConfiguration('output_height')
    camera_namespace = LaunchConfiguration('camera_namespace')

    argus_stereo_node = ComposableNode(
        name='argus_stereo',
        namespace=camera_namespace,
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusStereoNode',
        parameters=[{
            'module_id': module_id,
            'camera_id': camera_id,
            'mode': mode,
            'fsync_type': fsync_type,
            'use_hw_timestamp': use_hw_timestamp,
            'camera_link_frame_name': camera_link_frame_name,
            'left_camera_frame_name': left_camera_frame_name,
            'right_camera_frame_name': right_camera_frame_name,
            'left_camera_info_url': left_camera_info_url,
            'right_camera_info_url': right_camera_info_url,
            'wide_fov': wide_fov,
        }],
    )

    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        namespace=camera_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': output_width,
            'output_height': output_height,
        }],
        remappings=[
            ('image_raw', 'left/image_raw'),
            ('camera_info', 'left/camera_info_relay'),
            ('image_rect', 'left/image_rect'),
            ('camera_info_rect', 'left/camera_info_rect'),
        ],
    )

    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        namespace=camera_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': output_width,
            'output_height': output_height,
        }],
        remappings=[
            ('image_raw', 'right/image_raw'),
            ('camera_info', 'right/camera_info_relay'),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect'),
        ],
    )

    container = ComposableNodeContainer(
        name='argus_camera_container',
        namespace=camera_namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            argus_stereo_node,
            left_rectify_node,
            right_rectify_node,
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )

    left_camera_info_relay = Node(
        package='pycuvslam_ros',
        executable='camera_info_relay',
        name='left_camera_info_relay',
        namespace=camera_namespace,
        remappings=[
            ('image_raw', 'left/image_raw'),
            ('camera_info', 'left/camera_info'),
            ('camera_info_relay', 'left/camera_info_relay'),
        ],
        output='screen',
    )

    right_camera_info_relay = Node(
        package='pycuvslam_ros',
        executable='camera_info_relay',
        name='right_camera_info_relay',
        namespace=camera_namespace,
        remappings=[
            ('image_raw', 'right/image_raw'),
            ('camera_info', 'right/camera_info'),
            ('camera_info_relay', 'right/camera_info_relay'),
        ],
        output='screen',
    )

    left_image_compressor = Node(
        package='image_transport',
        executable='republish',
        name='left_image_compressor',
        namespace=camera_namespace,
        arguments=['raw', 'compressed'],
        remappings=[
            ('in', 'left/image_rect'),
            ('out/compressed', 'left/image_rect/compressed'),
        ],
        output='screen',
    )

    right_image_compressor = Node(
        package='image_transport',
        executable='republish',
        name='right_image_compressor',
        namespace=camera_namespace,
        arguments=['raw', 'compressed'],
        remappings=[
            ('in', 'right/image_rect'),
            ('out/compressed', 'right/image_rect/compressed'),
        ],
        output='screen',
    )

    return launch.LaunchDescription(
        launch_args + [
            container,
            left_camera_info_relay,
            right_camera_info_relay,
            left_image_compressor,
            right_image_compressor,
        ]
    )
