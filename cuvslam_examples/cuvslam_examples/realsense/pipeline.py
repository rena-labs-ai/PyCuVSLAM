import queue
from typing import Optional

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.tracker import BaseTracker


class Pipeline:
    """General tracking pipeline that delegates tracker-specific logic to a BaseTracker.

    Owns all shared state: result queue, vslam.Tracker instance, odometry config.
    Provides context-manager support and a uniform get() interface.
    """

    def __init__(self, inner: BaseTracker, queue_maxsize: int = 2) -> None:
        self._inner = inner
        self._queue: queue.Queue[TrackingResult] = queue.Queue(maxsize=queue_maxsize)
        self._tracker: Optional[vslam.Tracker] = None
        self._odometry_config: Optional[vslam.Tracker.OdometryConfig] = None

    def start(self) -> None:
        camera_params = self._inner.setup_camera_parameters()
        self._odometry_config = self._inner.create_odometry_config()
        rig = self._inner.create_rig(camera_params)
        slam_cfg = self._inner.create_slam_config()
        self._tracker = vslam.Tracker(rig, self._odometry_config, slam_cfg)
        self._inner.start_streaming(self._tracker, self._queue)

    def stop(self) -> None:
        self._inner.stop_streaming()

    def get(self, timeout: float = 1.0) -> Optional[TrackingResult]:
        try:
            return self._queue.get(timeout=timeout)
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
