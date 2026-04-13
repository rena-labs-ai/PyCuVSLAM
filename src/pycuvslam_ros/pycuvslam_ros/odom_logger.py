#!/usr/bin/env python3
"""Odometry trajectory logger and live plotter.

Subscribes to one reference odometry topic and one or more estimated odometry
topics, samples at 1 Hz, and periodically writes a combined trajectory PNG.

Parameters
----------
ref_odom_topic           : str   - reference topic (default: /Odometry)
estimated_odom_topics    : str   - comma-separated estimated topics
                                   (default: /cuvslam/odometry)
estimated_labels         : str   - comma-separated labels (default: topic names)
estimated_odom_rotations : str   - comma-separated rotation offsets in degrees
                                   applied to each estimated trajectory (default: all 0)
out_path                 : str   - output PNG path (default: ./odom_plot.png)
title                    : str   - plot title (default: "Odometry Comparison")
update_interval_sec      : float - seconds between plot refreshes (default: 2.0)
"""

import math
import sys
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry

from pycuvslam_ros.plot import compute_ate, plot_combined

_log = lambda msg: print(f"[odom_logger] {msg}", file=sys.stderr)


class OdomLogger(Node):
    def __init__(self) -> None:
        super().__init__("odom_logger")

        self.declare_parameter("ref_odom_topic", "/Odometry")
        self.declare_parameter("estimated_odom_topics", "/cuvslam/odometry")
        self.declare_parameter("estimated_labels", "")
        self.declare_parameter("out_path", "./odom_plot.png")
        self.declare_parameter("title", "Odometry Comparison")
        self.declare_parameter("update_interval_sec", 2.0)

        ref_topic = self.get_parameter("ref_odom_topic").get_parameter_value().string_value
        est_topics_raw = self.get_parameter("estimated_odom_topics").get_parameter_value().string_value
        labels_raw = self.get_parameter("estimated_labels").get_parameter_value().string_value
        self._out_path = Path(self.get_parameter("out_path").get_parameter_value().string_value)
        self._title = self.get_parameter("title").get_parameter_value().string_value
        update_interval = self.get_parameter("update_interval_sec").get_parameter_value().double_value

        est_topics = [t.strip() for t in est_topics_raw.split(",") if t.strip()]
        if not est_topics:
            raise ValueError("estimated_odom_topics must contain at least one topic")

        labels_list = [l.strip() for l in labels_raw.split(",") if l.strip()]
        self._labels = [
            labels_list[i] if i < len(labels_list) else t
            for i, t in enumerate(est_topics)
        ]

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._last_ref: Optional[Odometry] = None
        self._ref_traj: list[tuple] = []
        self._last_est: list[Optional[Odometry]] = [None] * len(est_topics)
        self._est_trajs: list[list[tuple]] = [[] for _ in est_topics]

        self.create_subscription(Odometry, ref_topic, self._on_ref, qos)
        for i, topic in enumerate(est_topics):
            self.create_subscription(
                Odometry, topic,
                lambda msg, idx=i: self._on_est(msg, idx),
                qos,
            )

        self.get_logger().info(f"ref={ref_topic}  estimated={est_topics}  labels={self._labels}")
        self.create_timer(1.0, self._sample)
        self.create_timer(update_interval, self._update_plot)

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _on_ref(self, msg: Odometry) -> None:
        self._last_ref = msg

    def _on_est(self, msg: Odometry, idx: int) -> None:
        self._last_est[idx] = msg

    def _sample(self) -> None:
        any_est = any(m is not None for m in self._last_est)
        if not any_est:
            return
        if self._last_ref is not None:
            r = self._last_ref.pose.pose
            self._ref_traj.append((r.position.x, r.position.y, self._quat_to_yaw(r.orientation)))
        for i, msg in enumerate(self._last_est):
            if msg is not None:
                c = msg.pose.pose
                self._est_trajs[i].append((c.position.x, c.position.y, self._quat_to_yaw(c.orientation)))

    def _update_plot(self) -> None:
        if not any(len(t) >= 2 for t in self._est_trajs):
            return
        try:
            estimated_list = list(zip(self._labels, self._est_trajs))
            plot_combined(self._ref_traj, estimated_list, self._title, self._out_path)
        except Exception as e:
            self.get_logger().warn(f"Plot update failed: {e}")

    def get_results(self):
        return self._ref_traj, list(zip(self._labels, self._est_trajs))


def _report(ref_traj, estimated_list) -> None:
    for label, est in estimated_list:
        n = len(est)
        if n < 2:
            _log(f"[{label}] Only {n} sample(s), skipping.")
            continue
        if ref_traj:
            ate_rmse, _ = compute_ate(ref_traj, est)
            _log(f"[{label}] ATE RMSE = {ate_rmse:.4f} m  ({min(len(ref_traj), n)} paired samples)")
        else:
            _log(f"[{label}] {n} samples (no reference).")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    ref_traj, estimated_list = node.get_results()

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

    # Final plot + report
    if any(len(est) >= 2 for _, est in estimated_list):
        try:
            out_path = node._out_path
            plot_combined(ref_traj, estimated_list, node._title, out_path)
            _log(f"Final plot saved to {out_path}")
        except Exception as e:
            _log(f"Failed to save final plot: {e}")
    _report(ref_traj, estimated_list)


if __name__ == "__main__":
    main()
