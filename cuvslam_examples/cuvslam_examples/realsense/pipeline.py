import queue
from typing import Callable, List, Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import cuvslam as vslam

from cuvslam_examples.realsense import TrackingResult
from cuvslam_examples.realsense.tracker import BaseTracker


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
        self._frame_id = 0

    def start(self) -> None:
        camera_params = self._inner.setup_camera_parameters()
        self._odometry_config = self._inner.create_odometry_config()
        rig = self._inner.create_rig(camera_params)
        for i, cam in enumerate(rig.cameras):
            print(
                f"  rig.cameras[{i}]: size={cam.size} focal={cam.focal} principal={cam.principal} "
                f"distortion.model={cam.distortion.model.name} distortion.parameters={cam.distortion.parameters} "
                f"rig_from_camera.translation={cam.rig_from_camera.translation} "
                f"rig_from_camera.rotation={cam.rig_from_camera.rotation} "
                f"borders=(t={cam.border_top},b={cam.border_bottom},l={cam.border_left},r={cam.border_right})"
            )
        if rig.imus:
            for j, imu in enumerate(rig.imus):
                print(
                    f"  rig.imus[{j}]: frequency={imu.frequency} rig_from_imu.translation={imu.rig_from_imu.translation} "
                    f"gyro_noise={imu.gyroscope_noise_density} accel_noise={imu.accelerometer_noise_density}"
                )
        slam_cfg = self._inner.create_slam_config()
        self._tracker = vslam.Tracker(rig, self._odometry_config, slam_cfg)
        if self._enable_visualization:
            self._init_rerun()
        kwargs = {}
        if self._get_latest_fast_lio is not None:
            kwargs["get_latest_fast_lio"] = self._get_latest_fast_lio
        self._inner.start_streaming(self._tracker, self._queue, **kwargs)

    def _init_rerun(self) -> None:
        rr.init("cuVSLAM Pipeline", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        num_cams = self._inner.num_viz_cameras
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.4, 0.6],
                    contents=[
                        rrb.Vertical(
                            contents=[
                                rrb.Spatial2DView(origin=f"world/camera_{i}")
                                for i in range(num_cams)
                            ]
                        ),
                        rrb.Spatial3DView(origin="world"),
                    ],
                ),
            ),
            make_active=True,
        )

    def _visualize(self, result: TrackingResult) -> None:
        if result.slam_pose is None:
            return
        self._frame_id += 1
        self._trajectory.append(np.array(result.slam_pose.translation))

        rr.set_time_sequence("frame", self._frame_id)
        rr.log("world/slam_trajectory", rr.LineStrips3D(self._trajectory), static=True)
        rr.log(
            "world/camera_0",
            rr.Transform3D(
                translation=result.slam_pose.translation,
                quaternion=result.slam_pose.rotation,
            ),
        )

        viz_idx = self._inner.get_viz_image_indices()
        for i, img_idx in enumerate(viz_idx):
            if img_idx < len(result.images):
                img = result.images[img_idx]
                if img.dtype == np.uint8:
                    rr.log(f"world/camera_{i}", rr.Image(img).compress())
                else:
                    rr.log(f"world/camera_{i}", rr.Image(img))

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
