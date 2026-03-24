"""ROS2 node that runs cuVSLAM and publishes odometry to /cuvslam/odometry.

Supports RosMulticamTracker (config_file) and RosZedStereoTracker (zed_left_topic, zed_right_topic).
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
from cuvslam_examples.realsense.tracker import RosMulticamTracker
from cuvslam_examples.zed.tracker import RosZedStereoTracker, ZedStereoTracker

ODOM_TOPIC = "/cuvslam/odometry"
ODOM_FRAME = "odom"
MAP_FRAME = "map"


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
    zed_left_param = param_node.declare_parameter(
        "zed_left_topic", "/zed/zed_node/left/color/rect/image/compressed"
    )
    zed_right_param = param_node.declare_parameter(
        "zed_right_topic", "/zed/zed_node/right/color/rect/image/compressed"
    )
    base_link_param = param_node.declare_parameter(
        "base_link_frame", "zed_camera_link"
    )

    config_file = config_file_param.value
    tracker_type = str(tracker_param.value)
    if tracker_type == "ros_multicam" and not config_file:
        param_node.get_logger().error("config_file parameter is required for ros_multicam")
        param_node.destroy_node()
        rclpy.shutdown()
        return

    enable_viz = enable_viz_param.value if isinstance(enable_viz_param.value, bool) else str(enable_viz_param.value).lower() == "true"
    param_node.destroy_node()

    if tracker_type == "ros_zed_stereo":
        tracker = RosZedStereoTracker(
            left_topic=str(zed_left_param.value),
            right_topic=str(zed_right_param.value),
        )
    elif tracker_type == "zed_stereo":
        tracker = ZedStereoTracker()
    else:
        tracker = RosMulticamTracker(config_file=config_file)

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
