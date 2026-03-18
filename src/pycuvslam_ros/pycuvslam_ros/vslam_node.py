"""ROS2 node that runs RosMulticamTracker and publishes odometry to /cuvslam/odometry."""

import threading
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseWithCovariance, Quaternion, TwistWithCovariance
from nav_msgs.msg import Odometry

from cuvslam_examples.realsense.pipeline import Pipeline
from cuvslam_examples.realsense.tracker import RosMulticamTracker

ODOM_TOPIC = "/cuvslam/odometry"
FRAME_ID = "cuvslam_init"
CHILD_FRAME_ID = "cuvslam_body"


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
    config_file = config_file_param.value
    if not config_file:
        param_node.get_logger().error("config_file parameter is required")
        param_node.destroy_node()
        rclpy.shutdown()
        return

    enable_viz = enable_viz_param.value if isinstance(enable_viz_param.value, bool) else str(enable_viz_param.value).lower() == "true"
    param_node.destroy_node()

    tracker = RosMulticamTracker(config_file=config_file)

    class VslamNode(Node):
        def __init__(self) -> None:
            super().__init__("vslam")
            self._odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
            self._pipeline = Pipeline(tracker, enable_visualization=enable_viz)
            self._pipeline.start()
            self.get_logger().info(f"Publishing odometry on {ODOM_TOPIC}")
            self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._thread.start()

        def _tracking_loop(self) -> None:
            while rclpy.ok():
                result = self._pipeline.get(timeout=1.0)
                if result is None or result.slam_pose is None:
                    continue
                stamp = _stamp_from_ns(result.timestamp)
                pose_robot = result.slam_pose.to_robot_frame()
                msg = Odometry()
                msg.header.stamp = stamp
                msg.header.frame_id = FRAME_ID
                msg.child_frame_id = CHILD_FRAME_ID
                t = pose_robot.translation
                r = pose_robot.rotation
                msg.pose = PoseWithCovariance()
                msg.pose.pose = Pose()
                msg.pose.pose.position = Point(x=float(t[0]), y=float(t[1]), z=float(t[2]))
                msg.pose.pose.orientation = Quaternion(x=float(r[0]), y=float(r[1]), z=float(r[2]), w=float(r[3]))
                msg.twist = TwistWithCovariance()
                self._odom_pub.publish(msg)

        def destroy_node(self) -> None:
            self._pipeline.stop()
            super().destroy_node()

    node = VslamNode()
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
