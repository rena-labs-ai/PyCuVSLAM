import argparse
from typing import List

import numpy as np
from numpy import array_equal as np_array_equal

import cuvslam as vslam

from cuvslam_examples.realsense.pipeline import Pipeline
from cuvslam_examples.realsense.tracker import BaseTracker, StereoTracker, VioTracker
from cuvslam_examples.realsense.utils import OdomLogger, Pose
from cuvslam_examples.realsense.visualizer import RerunVisualizer

TRACKERS = {
    "stereo": StereoTracker,
    "vio": VioTracker,
}


def run(tracker: BaseTracker) -> None:
    with Pipeline(tracker) as pipeline:
        visualizer = RerunVisualizer()
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

                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=[result.images[0]],
                    observations_main_cam=[pipeline.tracker.get_last_observations(0)],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="cuVSLAM tracking with visualization")
    parser.add_argument(
        "--tracker",
        choices=TRACKERS.keys(),
        default="stereo",
        help="Tracking mode (default: stereo)",
    )
    args = parser.parse_args()

    run(TRACKERS[args.tracker]())


if __name__ == "__main__":
    main()
