import argparse
import math
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy import array_equal as np_array_equal

import cuvslam as vslam

from cuvslam_examples.realsense.pipeline import Pipeline
from cuvslam_examples.realsense.tracker import (
    BaseTracker,
    MultiCamBagTracker,
    MultiCameraTracker,
    RGBDTracker,
    StereoTracker,
    VioTracker,
)
from cuvslam_examples.zed.tracker import RosZedStereoTracker, ZedStereoTracker
from cuvslam_examples.realsense.utils import OdomLogger, Pose
from cuvslam_examples.realsense.visualizer import RerunVisualizer

TRACKERS = {
    "stereo": StereoTracker,
    "stereo_zed": ZedStereoTracker,
    "ros_zed_stereo": RosZedStereoTracker,
    "vio": VioTracker,
    "multicam": MultiCameraTracker,
    "multicam_bag": MultiCamBagTracker,
    "rgbd": RGBDTracker,
}

GT_MATCH_THRESHOLD_NS = 100_000_000  # 100 ms


def _compute_max_diff(
    traj_a: List[tuple], traj_b: List[tuple], threshold_ns: float = GT_MATCH_THRESHOLD_NS
) -> tuple[float, int]:
    """Compute max horizontal diff between two (timestamp_ns, x, y, z) trajectories."""
    if not traj_a or not traj_b:
        return 0.0, 0
    bt = {ts: (x, y) for ts, x, y, _ in traj_b}
    b_ts = sorted(bt.keys())
    max_diff = 0.0
    max_idx = 0
    for idx, (ts, x, y, _) in enumerate(traj_a):
        i = np.searchsorted(b_ts, ts)
        if i >= len(b_ts):
            cand_ts = b_ts[-1]
        elif i == 0:
            cand_ts = b_ts[0]
        else:
            cand_ts = b_ts[i] if abs(b_ts[i] - ts) <= abs(b_ts[i - 1] - ts) else b_ts[i - 1]
        if abs(ts - cand_ts) <= threshold_ns:
            bx, by = bt[cand_ts]
            diff = math.sqrt((x - bx) ** 2 + (y - by) ** 2)
            if diff > max_diff:
                max_diff = diff
                max_idx = idx
    return max_diff, max_idx


def _plot_three_trajectories(
    ground_truth: List[tuple],
    vo_traj: List[tuple],
    slam_traj: List[tuple],
    title: str,
    out_path: Path,
    max_diff_vo: float,
    max_diff_slam: float,
) -> None:
    """Plot three trajectories (x, y) and write SVG."""
    all_pts = [(p[1], p[2]) for p in ground_truth + vo_traj + slam_traj]
    if not all_pts:
        raise ValueError("No trajectory data")

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    pad = 20
    w, h = 600, 600
    scale = min(
        (w - 2 * pad) / (max(xs) - min(xs) or 1),
        (h - 2 * pad) / (max(ys) - min(ys) or 1),
    )
    ox = pad - min(xs) * scale
    oy = pad + max(ys) * scale

    def path_str(pts: List[tuple]) -> str:
        if not pts:
            return ""
        scaled = [(ox + p[0] * scale, oy - p[1] * scale) for p in pts]
        return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in scaled)

    gt_xy = [(p[1], p[2]) for p in sorted(ground_truth, key=lambda x: x[0])]
    vo_xy = [(p[1], p[2]) for p in sorted(vo_traj, key=lambda x: x[0])]
    slam_xy = [(p[1], p[2]) for p in sorted(slam_traj, key=lambda x: x[0])]

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
    svg += f'<text x="10" y="20" font-size="14">{title}</text>'
    svg += f'<line x1="0" y1="{oy}" x2="{w}" y2="{oy}" stroke="#ccc" stroke-width="1"/>'
    svg += f'<line x1="{ox}" y1="0" x2="{ox}" y2="{h}" stroke="#ccc" stroke-width="1"/>'
    if gt_xy:
        svg += f'<path d="{path_str(gt_xy)}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
    if vo_xy:
        svg += f'<path d="{path_str(vo_xy)}" fill="none" stroke="#2ca02c" stroke-width="1.5" stroke-dasharray="4,4"/>'
    if slam_xy:
        svg += f'<path d="{path_str(slam_xy)}" fill="none" stroke="#ff7f0e" stroke-width="1.5" stroke-dasharray="8,4"/>'
    legend_y = h - 50
    svg += f'<text x="10" y="{legend_y}" font-size="12" fill="#1f77b4">— ground_truth</text>'
    svg += f'<text x="10" y="{legend_y + 14}" font-size="12" fill="#2ca02c">- - VO (max_diff={max_diff_vo:.4f}m)</text>'
    svg += f'<text x="10" y="{legend_y + 28}" font-size="12" fill="#ff7f0e">- - SLAM (max_diff={max_diff_slam:.4f}m)</text>'
    svg += "</svg>"
    out_path.write_text(svg)


def run(tracker: BaseTracker) -> None:
    with Pipeline(tracker) as pipeline:
        viz_img_idx = tracker.get_viz_image_indices()
        viz_obs_idx = tracker.get_viz_observation_indices()
        visualizer = RerunVisualizer(num_viz_cameras=tracker.num_viz_cameras)
        odom_logger = OdomLogger()

        frame_id = 0
        slam_trajectory: List[np.ndarray] = []
        loop_closure_poses: List[np.ndarray] = []

        try:
            while True:
                result = pipeline.get(timeout=1.0)
                if result is None:
                    continue

                if result.vo_pose is None or result.slam_pose is None:
                    continue

                frame_id += 1
                slam_trajectory.append(result.slam_pose.translation)

                odom_logger.log(frame_id, Pose(result.vo_pose), Pose(result.slam_pose))

                gravity = None
                if (
                    hasattr(pipeline.odometry_config, "odometry_mode")
                    and pipeline.odometry_config.odometry_mode
                    == vslam.Tracker.OdometryMode.Inertial
                ):
                    gravity = pipeline.tracker.get_last_gravity()

                current_lc_poses = pipeline.tracker.get_loop_closure_poses()
                if current_lc_poses and (
                    not loop_closure_poses
                    or not np_array_equal(
                        current_lc_poses[-1].pose.translation, loop_closure_poses[-1]
                    )
                ):
                    loop_closure_poses.append(current_lc_poses[-1].pose.translation)

                images = [result.images[i] for i in viz_img_idx]
                observations = [
                    pipeline.tracker.get_last_observations(i) for i in viz_obs_idx
                ]

                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=images,
                    observations_main_cam=observations,
                    slam_pose=result.slam_pose,
                    slam_trajectory=slam_trajectory,
                    timestamp=result.timestamp,
                    gravity=gravity,
                    final_landmarks=pipeline.tracker.get_final_landmarks(),
                    loop_closure_poses=(
                        loop_closure_poses if loop_closure_poses else None
                    ),
                )

        except KeyboardInterrupt:
            print("Stopping tracking...")
        finally:
            odom_logger.close()


def run_bag(
    tracker: BaseTracker,
    bag_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    name: str = "bag",
) -> None:
    """Run bag tracker (multicam_bag), collect trajectories, plot on exit.
    If bag_path is None, assumes bag is already playing (e.g. ros2 bag play in another terminal)."""
    import rclpy

    rclpy.init()
    bag_proc: Optional[subprocess.Popen] = None
    if bag_path is not None:
        bag_path = Path(bag_path)
        if not bag_path.exists():
            raise FileNotFoundError(f"Bag path not found: {bag_path}")
        print(f"[{name}] Playing bag: {bag_path}")
        bag_proc = subprocess.Popen(
            ["ros2", "bag", "play", str(bag_path), "--rate", "0.5"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(2.0)
        if bag_proc.poll() is not None:
            _, err = bag_proc.communicate()
            raise RuntimeError(f"ros2 bag play failed: {err.decode() if err else 'unknown'}")
    else:
        print(f"[{name}] No --bag; assuming bag already playing.")

    print(f"[{name}] Starting pipeline...")
    vo_traj: List[tuple] = []
    slam_traj: List[tuple] = []

    try:
        with Pipeline(tracker) as pipeline:
            try:
                while True:
                    if bag_proc is not None and bag_proc.poll() is not None:
                        print(f"[{name}] Bag finished, draining...")
                        for _ in range(10):
                            result = pipeline.get(timeout=0.3)
                            if result is None:
                                break
                            if result.vo_pose and result.slam_pose:
                                vo_rf = result.vo_pose.to_robot_frame()
                                slam_rf = result.slam_pose.to_robot_frame()
                                ts = result.timestamp
                                vo_traj.append((ts, vo_rf.translation[0], vo_rf.translation[1], vo_rf.translation[2]))
                                slam_traj.append((ts, slam_rf.translation[0], slam_rf.translation[1], slam_rf.translation[2]))
                        break
                    result = pipeline.get(timeout=0.5)
                    if result is None:
                        continue
                    if result.vo_pose is None or result.slam_pose is None:
                        continue

                    vo_rf = result.vo_pose.to_robot_frame()
                    slam_rf = result.slam_pose.to_robot_frame()
                    ts = result.timestamp
                    vo_traj.append((ts, vo_rf.translation[0], vo_rf.translation[1], vo_rf.translation[2]))
                    slam_traj.append((ts, slam_rf.translation[0], slam_rf.translation[1], slam_rf.translation[2]))
            except KeyboardInterrupt:
                print("Stopping tracking...")
                if bag_proc is not None:
                    bag_proc.terminate()

        gt_traj = tracker.get_ground_truth_trajectory()
        print(f"[{name}] Collected: {len(vo_traj)} VO, {len(slam_traj)} SLAM, {len(gt_traj)} ground_truth")

        all_pts = [(p[1], p[2]) for p in gt_traj + vo_traj + slam_traj]
        if not all_pts:
            print(f"[{name}] No trajectory data to plot. Check: bag has image topics, sync works, ground_truth_topic matches (e.g. /chassis/odom for Galileo).")
        else:
            max_diff_vo, _ = _compute_max_diff(vo_traj, gt_traj)
            max_diff_slam, _ = _compute_max_diff(slam_traj, gt_traj)
            print(f"[{name}] max_diff VO: {max_diff_vo:.4f} m | SLAM: {max_diff_slam:.4f} m")

            out = output_path or Path(f"{name}_trajectories.svg")
            _plot_three_trajectories(
                gt_traj, vo_traj, slam_traj,
                name,
                out,
                max_diff_vo,
                max_diff_slam,
            )
            print(f"[{name}] Saved to {out}")
    finally:
        if bag_proc is not None and bag_proc.poll() is None:
            bag_proc.terminate()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="cuVSLAM tracking with visualization")
    parser.add_argument(
        "--tracker",
        choices=TRACKERS.keys(),
        default="stereo",
        help="Tracking mode (default: stereo)",
    )
    parser.add_argument(
        "--bag",
        type=Path,
        help="ROS bag path (optional; if omitted, bag must already be playing)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output SVG path (bag trackers)")
    parser.add_argument(
        "--config",
        type=Path,
        help="Rig config YAML for multicam_bag (default: frame_agx_rig.yaml)",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        help="Filter multicam_bag by camera names (e.g. front left right)",
    )
    parser.add_argument(
        "--serial-numbers",
        nargs="+",
        help="Filter multicam_bag by RealSense serial numbers",
    )
    parser.add_argument(
        "--camera-topic-prefix",
        default="camera",
        help="Topic prefix for RealSense bags (default: camera)",
    )
    parser.add_argument(
        "--left-ir-topic",
        default="infra1/image_rect_raw/compressed",
        help="Left image topic relative to camera base (default: infra1/...)",
    )
    parser.add_argument(
        "--right-ir-topic",
        default="infra2/image_rect_raw/compressed",
        help="Right image topic relative to camera base (default: infra2/...)",
    )
    parser.add_argument(
        "--ground-truth-topic",
        default="/Odometry",
        help="Ground truth odometry topic (default: /Odometry; use /chassis/odom for Galileo)",
    )
    parser.add_argument(
        "--sync-slop",
        type=float,
        default=0.2,
        help="Sync slop in seconds for ApproximateTimeSynchronizer (default: 0.2)",
    )
    parser.add_argument(
        "--zed-left-topic",
        default="/zed/zed_node/left/color/rect/image/compressed",
        help="ZED left image topic for ros_zed_stereo (default: /zed/zed_node/left/color/rect/image/compressed)",
    )
    parser.add_argument(
        "--zed-right-topic",
        default="/zed/zed_node/right/color/rect/image/compressed",
        help="ZED right image topic for ros_zed_stereo (default: /zed/zed_node/right/color/rect/image/compressed)",
    )
    args = parser.parse_args()

    tracker_cls = TRACKERS[args.tracker]
    if args.tracker == "ros_zed_stereo":
        kwargs = {
            "left_topic": args.zed_left_topic,
            "right_topic": args.zed_right_topic,
        }
        tracker = tracker_cls(**kwargs)
    elif args.tracker == "multicam_bag":
        kwargs = {
            "serial_numbers": args.serial_numbers,
            "camera_names": args.camera_names,
            "camera_topic_prefix": args.camera_topic_prefix,
            "left_ir_topic": args.left_ir_topic,
            "right_ir_topic": args.right_ir_topic,
            "ground_truth_topic": args.ground_truth_topic,
            "sync_slop": args.sync_slop,
        }
        if args.config is not None:
            kwargs["config_file"] = str(args.config)
        tracker = tracker_cls(**kwargs)
    else:
        tracker = tracker_cls()

    if args.tracker == "multicam_bag":
        run_bag(tracker, args.bag, args.output, args.tracker)
    else:
        run(tracker)


if __name__ == "__main__":
    main()
