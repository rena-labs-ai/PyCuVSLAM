#!/usr/bin/env python3
"""Log odom x,y from two odometry topics every 1s, compare drift.

Computes standard SLAM evaluation metrics on shutdown:
- ATE (Absolute Trajectory Error): RMSE of pointwise Euclidean distances
- RPE (Relative Pose Error): RMSE of relative displacement errors over delta steps
"""

import math
import sys
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry

from pycuvslam_ros.plot import compute_ate, compute_rpe, compute_rpe_rotation, plot

_log = lambda msg: print(f"[odom_diff_logger] {msg}", file=sys.stderr)


class OdomDiffLogger(Node):
    def __init__(self) -> None:
        super().__init__("odom_diff_logger")
        self.declare_parameter("experiment", "default")
        self.declare_parameter("log_dir", "/tmp")
        self.declare_parameter("ref_odom_topic", "/Odometry")
        self.declare_parameter("other_odom_topic", "/cuvslam/odometry")
        exp = self.get_parameter("experiment").get_parameter_value().string_value
        self._log_dir = Path(self.get_parameter("log_dir").get_parameter_value().string_value)
        ref_topic = self.get_parameter("ref_odom_topic").get_parameter_value().string_value
        other_topic = self.get_parameter("other_odom_topic").get_parameter_value().string_value
        self._experiment = exp
        self._last_ref: Optional[Odometry] = None
        self._last_other: Optional[Odometry] = None
        self._ref_trajectory: list[tuple[float, float, float]] = []
        self._other_trajectory: list[tuple[float, float, float]] = []
        self._jitter_points: list[tuple[float, float]] = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Odometry, ref_topic, self._on_ref, qos)
        self.create_subscription(Odometry, other_topic, self._on_other, qos)
        self.create_subscription(PointStamped, "/cuvslam/jitter", self._on_jitter, 10)
        self.get_logger().info(f"Comparing '{ref_topic}' vs '{other_topic}'")

        self.create_timer(1.0, self._sample)
        self.create_timer(2.0, self._update_plot)

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _on_ref(self, msg: Odometry) -> None:
        self._last_ref = msg

    def _on_other(self, msg: Odometry) -> None:
        self._last_other = msg

    def _on_jitter(self, msg: PointStamped) -> None:
        self._jitter_points.append((msg.point.x, msg.point.y))

    def _sample(self) -> None:
        if self._last_other is None:
            return
        c = self._last_other.pose.pose
        self._other_trajectory.append((c.position.x, c.position.y, self._quat_to_yaw(c.orientation)))
        if self._last_ref is not None:
            r = self._last_ref.pose.pose
            self._ref_trajectory.append((r.position.x, r.position.y, self._quat_to_yaw(r.orientation)))

    def _out_path(self) -> Path:
        out_dir = self._log_dir / "experiments"
        out_dir.mkdir(exist_ok=True)
        return out_dir / f"{self._experiment}.png"

    def _update_plot(self) -> None:
        if len(self._other_trajectory) < 2:
            return
        n = min(len(self._ref_trajectory), len(self._other_trajectory)) if self._ref_trajectory else 0
        ref = self._ref_trajectory[:n] if n else []
        other = self._other_trajectory[-n:] if n else list(self._other_trajectory)
        try:
            metrics = _compute_metrics(ref, other) if ref else None
            plot(ref, other, self._experiment, self._out_path(), metrics=metrics,
                 jitter_points=list(self._jitter_points))
        except Exception as e:
            self.get_logger().warn(f"Plot update failed: {e}")

    def get_results(self) -> tuple[list, list, list, str, Path]:
        n = min(len(self._ref_trajectory), len(self._other_trajectory)) if self._ref_trajectory else 0
        ref = self._ref_trajectory[:n] if n else []
        other = self._other_trajectory[-n:] if n else list(self._other_trajectory)
        return ref, other, list(self._jitter_points), self._experiment, self._log_dir


def _compute_metrics(ref, other) -> dict[str, float]:
    ate_rmse, ate_mean = compute_ate(ref, other)
    rpe_rmse, rpe_mean = compute_rpe(ref, other, delta=1)
    rpe_rot_rmse, rpe_rot_mean = compute_rpe_rotation(ref, other, delta=1)
    return {
        "ate_rmse": ate_rmse, "ate_mean": ate_mean,
        "rpe_rmse": rpe_rmse, "rpe_mean": rpe_mean,
        "rpe_rot_rmse": rpe_rot_rmse, "rpe_rot_mean": rpe_rot_mean,
    }


def _save_results(ref, other, jitter_points, experiment, log_dir) -> None:
    n = len(other)
    if n < 2:
        _log(f"Only {n} sample(s) collected, skipping plot.")
        return

    metrics = _compute_metrics(ref, other) if ref else None
    if metrics:
        _log(
            f"ATE RMSE={metrics['ate_rmse']:.4f}m  "
            f"RPE RMSE={metrics['rpe_rmse']:.4f}m  "
            f"RPE Rot RMSE={math.degrees(metrics['rpe_rot_rmse']):.4f}deg  "
            f"({min(len(ref), len(other))} paired samples)"
        )
    else:
        _log(f"No ground truth available. Plotting {n} cuvslam-only samples.")

    try:
        out_dir = log_dir / "experiments"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{experiment}.png"
        plot(ref, other, experiment, out_path, metrics=metrics if metrics else None,
             jitter_points=jitter_points)
        _log(f"Saved trajectory plot to {out_path}")
    except Exception as e:
        _log(f"Failed to generate trajectory plot: {e}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomDiffLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    ref, other, jitter_points, experiment, log_dir = node.get_results()

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

    _save_results(ref, other, jitter_points, experiment, log_dir)


if __name__ == "__main__":
    main()
