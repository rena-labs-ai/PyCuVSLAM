"""ROS2 node that runs cuVSLAM and publishes odometry to /cuvslam/odometry.

Supports RosMulticamTracker, RosZedStereoTracker, RosZedVIOTracker,
RosRealSenseStereoTracker, RosHawkStereoTracker, and RosHawkMulticamTracker.
Stereo image topics are derived from parameter camera (model id).
Publishes TF: map->odom (identity), odom->base_link (from odometry).
"""

import threading
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseWithCovariance, Quaternion, TransformStamped, TwistWithCovariance, Vector3
from nav_msgs.msg import Odometry
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from cuvslam_examples.realsense.pipeline import Pipeline
from cuvslam_examples.realsense.tracker import RosMulticamTracker, RosRealSenseStereoTracker
from cuvslam_examples.zed.tracker import RosZedStereoTracker, RosZedVIOTracker
from cuvslam_examples.hawk.tracker import RosHawkMulticamTracker, RosHawkStereoTracker

ODOM_TOPIC = "/cuvslam/odometry"
ODOM_FRAME = "odom"
MAP_FRAME = "map"

# Default ROS topic namespace (prefix) per RealSense camera model.
_REALSENSE_TOPIC_BASE_DEFAULT = {
    "realsensed435": "/realsense435_base",
    "realsensed455": "/realsense455_base",
}


def _zed_stem(camera: str) -> str:
    c = camera.lower().strip()
    if c in ("zed", "zedm", "zed2i"):
        return f"/{c}_base/zed_node"
    return "/zed/zed_node"


def _realsense_topic_base(camera: str) -> str:
    c = camera.lower().strip()
    if c not in _REALSENSE_TOPIC_BASE_DEFAULT:
        raise ValueError(
            f"Unknown RealSense camera model {camera!r}; expected one of "
            f"{sorted(_REALSENSE_TOPIC_BASE_DEFAULT)}"
        )
    return _REALSENSE_TOPIC_BASE_DEFAULT[c]


def _stamp_from_ns(timestamp_ns: int) -> Time:
    t = Time()
    t.sec = int(timestamp_ns // 1_000_000_000)
    t.nanosec = int(timestamp_ns % 1_000_000_000)
    return t


def main() -> None:
    rclpy.init()

    param_node = Node("vslam")
    config_file_param = param_node.declare_parameter("config_file", "")
    enable_viz_param = param_node.declare_parameter("enable_visualization", False)
    tracker_param = param_node.declare_parameter("tracker", "ros_multicam")
    camera_param = param_node.declare_parameter("camera", "zed2i")
    hawk_rig_param = param_node.declare_parameter(
        "hawk_rig_file", ""
    )
    base_link_param = param_node.declare_parameter(
        "base_link_frame", "zed_camera_link"
    )

    config_file = config_file_param.value
    tracker_type = str(tracker_param.value)
    hawk_rig_file = str(hawk_rig_param.value)
    camera_model = str(camera_param.value)

    if tracker_type == "ros_multicam" and not config_file:
        param_node.get_logger().error("config_file parameter is required for ros_multicam")
        param_node.destroy_node()
        rclpy.shutdown()
        return
    if tracker_type == "ros_hawk_multicam" and not hawk_rig_file:
        param_node.get_logger().error("hawk_rig_file parameter is required for ros_hawk_multicam")
        param_node.destroy_node()
        rclpy.shutdown()
        return

    enable_viz = enable_viz_param.value if isinstance(enable_viz_param.value, bool) else str(enable_viz_param.value).lower() == "true"
    param_node.destroy_node()

    zed_stem = _zed_stem(camera_model)
    zed_left = f"{zed_stem}/left/color/rect/image/compressed"
    zed_right = f"{zed_stem}/right/color/rect/image/compressed"
    zed_imu = f"{zed_stem}/imu/data"
    hawk_left = "/left/image_rect"
    hawk_right = "/right/image_rect"

    match tracker_type:
        case "ros_hawk_stereo":
            tracker = RosHawkStereoTracker(left_topic=hawk_left, right_topic=hawk_right)
        case "ros_hawk_multicam":
            tracker = RosHawkMulticamTracker(rig_file=hawk_rig_file)
        case "ros_zed_stereo":
            tracker = RosZedStereoTracker(left_topic=zed_left, right_topic=zed_right)
        case "ros_zed_vio":
            tracker = RosZedVIOTracker(
                left_topic=zed_left,
                right_topic=zed_right,
                imu_topic=zed_imu,
            )
        case "ros_realsense_stereo":
            tracker = RosRealSenseStereoTracker(
                topic_base=_realsense_topic_base(camera_model),
            )
        case _:
            raise ValueError(f"Unknown tracker type: {tracker_type}")

    base_link_frame = str(base_link_param.value)

    class VslamNode(Node):
        def __init__(self, child_frame: str) -> None:
            super().__init__("vslam")
            self._child_frame = child_frame
            self._odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
            self._tf_broadcaster = TransformBroadcaster(self)
            self._static_tf_broadcaster = StaticTransformBroadcaster(self)
            self._pipeline = Pipeline(tracker, enable_visualization=enable_viz)
            self._pipeline.start()

            # map -> odom: identity (no localization yet)
            static_tf = TransformStamped()
            static_tf.header.stamp = self.get_clock().now().to_msg()
            static_tf.header.frame_id = MAP_FRAME
            static_tf.child_frame_id = ODOM_FRAME
            static_tf.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
            static_tf.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            self._static_tf_broadcaster.sendTransform(static_tf)

            self.get_logger().info(f"Publishing odometry on {ODOM_TOPIC}, TF odom->{child_frame}")
            self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._thread.start()

        def _tracking_loop(self) -> None:
            while rclpy.ok():
                result = self._pipeline.get(timeout=1.0)
                if result is None or result.slam_pose is None:
                    continue
                stamp = _stamp_from_ns(result.timestamp)
                pose_robot = result.slam_pose.to_robot_frame()
                t = pose_robot.translation
                r = pose_robot.rotation

                msg = Odometry()
                msg.header.stamp = stamp
                msg.header.frame_id = ODOM_FRAME
                msg.child_frame_id = self._child_frame
                msg.pose = PoseWithCovariance()
                msg.pose.pose = Pose()
                msg.pose.pose.position = Point(x=float(t[0]), y=float(t[1]), z=float(t[2]))
                msg.pose.pose.orientation = Quaternion(x=float(r[0]), y=float(r[1]), z=float(r[2]), w=float(r[3]))
                msg.twist = TwistWithCovariance()
                self._odom_pub.publish(msg)

                # odom -> base_link
                tf = TransformStamped()
                tf.header.stamp = stamp
                tf.header.frame_id = ODOM_FRAME
                tf.child_frame_id = self._child_frame
                tf.transform.translation = Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2]))
                tf.transform.rotation = Quaternion(x=float(r[0]), y=float(r[1]), z=float(r[2]), w=float(r[3]))
                self._tf_broadcaster.sendTransform(tf)

        def destroy_node(self) -> None:
            self._pipeline.stop()
            super().destroy_node()

    node = VslamNode(child_frame=base_link_frame)
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
