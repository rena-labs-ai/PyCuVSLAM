"""Relay node that re-stamps camera_info with every incoming image_raw timestamp.

The isaac_ros RectifyNode uses a GXF Synchronization component that requires
exact timestamp matches between image_raw and camera_info. ArgusStereoNode
publishes camera_info with irregular timing, causing the synchronizer to drop
frames. This node caches the first camera_info it receives, then re-publishes
it with the timestamp of every image_raw frame so the synchronizer always finds
a pair.

Usage (per camera):
  camera_info_relay image_raw:=left/image_raw \
                    camera_info:=left/camera_info \
                    camera_info_relay:=left/camera_info_relay
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
import copy


class CameraInfoRelayNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_info_relay")

        self._cached_info: CameraInfo | None = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub = self.create_publisher(CameraInfo, "camera_info_relay", reliable_qos)

        self.create_subscription(
            CameraInfo, "camera_info", self._on_camera_info, reliable_qos
        )
        self.create_subscription(
            Image, "image_raw", self._on_image, sensor_qos
        )

        self.get_logger().info("camera_info_relay: waiting for first CameraInfo...")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._cached_info is None:
            self._cached_info = msg
            self.get_logger().info("camera_info_relay: CameraInfo cached.")

    def _on_image(self, msg: Image) -> None:
        if self._cached_info is None:
            return
        out = copy.copy(self._cached_info)
        out.header.stamp = msg.header.stamp
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = CameraInfoRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
