#!/usr/bin/env python3
"""Subscribe to IR camera topics and log Hz per camera: cam_1: 30, cam_2: 30, ..."""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class CameraHzLogger(Node):
    def __init__(self) -> None:
        super().__init__("camera_hz_logger")
        self.declare_parameter(
            "camera_topics",
            "/front/camera/infra1/image_rect_raw /front/camera/infra2/image_rect_raw "
            "/left/camera/infra1/image_rect_raw /left/camera/infra2/image_rect_raw "
            "/right/camera/infra1/image_rect_raw /right/camera/infra2/image_rect_raw "
            "/back/camera/infra1/image_rect_raw /back/camera/infra2/image_rect_raw",
        )
        topics_raw = self.get_parameter("camera_topics").get_parameter_value().string_value
        topics = [t.strip() for t in topics_raw.split() if t.strip()]

        self._counts: dict[str, int] = {topic: 0 for topic in topics}
        self._start_time = time.monotonic()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        for topic in topics:
            msg_type = CompressedImage if topic.endswith("/compressed") else Image
            self.create_subscription(msg_type, topic, self._make_callback(topic), qos_profile=qos)

        self.create_timer(1.0, self._log_hz)
        self.get_logger().info(f"Subscribed to {len(topics)} topics: {topics}")

    def _make_callback(self, topic: str):
        def cb(_msg) -> None:
            self._counts[topic] += 1

        return cb

    def _log_hz(self) -> None:
        elapsed = time.monotonic() - self._start_time
        if elapsed < 0.01:
            return
        parts = []
        for i, (topic, count) in enumerate(self._counts.items(), 1):
            hz = count / elapsed
            parts.append(f"cam_{i}: {hz:.1f}")
            self._counts[topic] = 0
        self._start_time = time.monotonic()
        self.get_logger().info(", ".join(parts))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraHzLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
