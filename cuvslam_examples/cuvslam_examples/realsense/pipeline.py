import queue
from typing import Callable, List, Optional

import numpy as np
from numpy import array_equal as np_array_equal

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.tracker import BaseTracker
from cuvslam_examples.realsense.visualizer import RerunVisualizer


class Pipeline:
    """General tracking pipeline that delegates tracker-specific logic to a BaseTracker.

    Owns all shared state: result queue, vslam.Tracker instance, odometry config.
    Provides context-manager support and a uniform get() interface.
    """

    def __init__(
        self,
        inner: BaseTracker,
        queue_maxsize: int = 2,
        enable_visualization: bool = False,
        get_latest_fast_lio: Optional[Callable[[], Optional[object]]] = None,
    ) -> None:
        self._inner = inner
        self._queue: queue.Queue[TrackingResult] = queue.Queue(maxsize=queue_maxsize)
        self._tracker: Optional[vslam.Tracker] = None
        self._odometry_config: Optional[vslam.Tracker.OdometryConfig] = None
        self._enable_visualization = enable_visualization
        self._get_latest_fast_lio = get_latest_fast_lio
        self._trajectory: List[np.ndarray] = []
        self._loop_closure_poses: List[np.ndarray] = []
        self._frame_id = 0
        self._visualizer: Optional[RerunVisualizer] = None

    def start(self) -> None:
        camera_params = self._inner.setup_camera_parameters()
        self._odometry_config = self._inner.create_odometry_config()
        rig = self._inner.create_rig(camera_params)
        print(rig)
        slam_cfg = self._inner.create_slam_config()
        self._tracker = vslam.Tracker(rig, self._odometry_config, slam_cfg)
        if self._enable_visualization:
            self._visualizer = RerunVisualizer(num_viz_cameras=self._inner.num_viz_cameras)
        kwargs = {}
        if self._get_latest_fast_lio is not None:
            kwargs["get_latest_fast_lio"] = self._get_latest_fast_lio
        self._inner.start_streaming(self._tracker, self._queue, **kwargs)

    def _visualize(self, result: TrackingResult) -> None:
        if result.slam_pose is None or self._visualizer is None:
            return
        self._frame_id += 1
        self._trajectory.append(np.array(result.slam_pose.translation))

        current_lc = self._tracker.get_loop_closure_poses()
        if current_lc and (
            not self._loop_closure_poses
            or not np_array_equal(
                current_lc[-1].pose.translation, self._loop_closure_poses[-1]
            )
        ):
            self._loop_closure_poses.append(current_lc[-1].pose.translation)

        gravity = None
        if (
            hasattr(self._odometry_config, "odometry_mode")
            and self._odometry_config.odometry_mode
            == vslam.Tracker.OdometryMode.Inertial
        ):
            gravity = self._tracker.get_last_gravity()

        viz_img_idx = self._inner.get_viz_image_indices()
        viz_obs_idx = self._inner.get_viz_observation_indices()
        images = [result.images[i] for i in viz_img_idx]
        observations = [self._tracker.get_last_observations(i) for i in viz_obs_idx]

        self._visualizer.visualize_frame(
            frame_id=self._frame_id,
            images=images,
            observations_main_cam=observations,
            slam_pose=result.slam_pose,
            slam_trajectory=self._trajectory,
            timestamp=result.timestamp,
            gravity=gravity,
            final_landmarks=self._tracker.get_final_landmarks(),
            loop_closure_poses=(
                self._loop_closure_poses if self._loop_closure_poses else None
            ),
        )

    def stop(self) -> None:
        self._inner.stop_streaming()

    def get(self, timeout: float = 1.0) -> Optional[TrackingResult]:
        try:
            result = self._queue.get(timeout=timeout)
            if result and self._enable_visualization:
                self._visualize(result)
            return result
        except queue.Empty:
            return None

    @property
    def tracker(self) -> vslam.Tracker:
        return self._tracker

    @property
    def odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return self._odometry_config

    def __enter__(self) -> "Pipeline":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
