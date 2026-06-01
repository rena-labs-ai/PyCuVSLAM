"""ROS2 node that runs cuVSLAM on an OAK camera and publishes odometry to
/cuvslam/odometry.

Supports the OAK trackers: RosOakStereoTracker and RosOakRGBDTracker. Stereo
image topics are derived inside the tracker from rena_bringup/config/config.yaml
(serial + image_mode -> image_raw | image_rect).

Publishes only Odometry (no TF). Stamps match the tracker pipeline timestamp
(synced left IR header time from ApproximateTimeSynchronizer).
"""

import threading
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    Quaternion,
    TwistWithCovariance,
)
from nav_msgs.msg import Odometry

# Pipeline is the shared tracking infrastructure (not RealSense-specific); it
# lives under cuvslam_examples.realsense and is reused by the OAK trackers.
from cuvslam_examples.realsense.pipeline import Pipeline
from cuvslam_examples.oak.tracker import RosOakRGBDTracker, RosOakStereoTracker

ODOM_TOPIC = "/cuvslam/odometry"
ODOM_FRAME = "odom"


def _stamp_from_ns(timestamp_ns: int) -> Time:
    t = Time()
    t.sec = int(timestamp_ns // 1_000_000_000)
    t.nanosec = int(timestamp_ns % 1_000_000_000)
    return t


def main() -> None:
    rclpy.init()

    param_node = Node("vslam")
    enable_viz_param = param_node.declare_parameter("enable_visualization", False)
    tracker_param = param_node.declare_parameter("tracker", "ros_oak_rgbd")
    odom_child_frame_param = param_node.declare_parameter(
        "odom_child_frame", "nav_base_link"
    )

    tracker_type = str(tracker_param.value)
    odom_child_frame = str(odom_child_frame_param.value)

    enable_viz = (
        enable_viz_param.value
        if isinstance(enable_viz_param.value, bool)
        else str(enable_viz_param.value).lower() == "true"
    )
    param_node.destroy_node()

    # OAK topics are derived from rena_bringup/config/config.yaml inside the
    # tracker itself (serial + image_mode -> image_raw | image_rect).
    match tracker_type:
        case "ros_oak_stereo":
            tracker = RosOakStereoTracker()
        case "ros_oak_rgbd":
            tracker = RosOakRGBDTracker()
        case _:
            raise ValueError(
                f"Unknown tracker type: {tracker_type!r}; "
                "supported: 'ros_oak_stereo', 'ros_oak_rgbd'"
            )

    class VslamNode(Node):
        def __init__(self, child_frame: str) -> None:
            super().__init__("vslam")
            self._child_frame = child_frame
            self._odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
            self._pipeline = Pipeline(tracker, enable_visualization=enable_viz)
            self._pipeline.start()

            self.get_logger().info(
                f"Publishing odometry only on {ODOM_TOPIC} "
                f"(child_frame={child_frame}, no TF)"
            )
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
                msg.pose.pose.position = Point(
                    x=float(t[0]), y=float(t[1]), z=float(t[2])
                )
                msg.pose.pose.orientation = Quaternion(
                    x=float(r[0]), y=float(r[1]), z=float(r[2]), w=float(r[3])
                )
                msg.twist = TwistWithCovariance()
                self._odom_pub.publish(msg)

        def destroy_node(self) -> None:
            self._pipeline.stop()
            super().destroy_node()

    node = VslamNode(child_frame=odom_child_frame)
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
