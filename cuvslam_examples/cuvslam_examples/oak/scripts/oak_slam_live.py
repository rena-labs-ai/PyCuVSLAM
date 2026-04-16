#!/usr/bin/env python3
"""Live OAK-D stereo cuVSLAM — plots trajectory on Ctrl+C."""

import signal
import sys
from datetime import timedelta
from typing import List, Optional

import depthai as dai
import matplotlib.pyplot as plt
import numpy as np

import cuvslam as vslam

FPS = 30
RESOLUTION = (1280, 800)
WARMUP_FRAMES = 60
CM_TO_METERS = 100


def _oak_extrinsics_to_pose(mat: List[List[float]]) -> vslam.Pose:
    m = np.array(mat)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(m[:3, :3]).as_quat()
    t = m[:3, 3] / CM_TO_METERS
    return vslam.Pose(rotation=q, translation=t)


def _build_camera(calib: dai.CalibrationHandler, socket: dai.CameraBoardSocket,
                  resolution, ref_socket: dai.CameraBoardSocket) -> vslam.Camera:
    cam = vslam.Camera()
    intr = calib.getCameraIntrinsics(socket, resolution[0], resolution[1])
    dist = calib.getDistortionCoefficients(socket)[:8]
    extr = calib.getCameraExtrinsics(socket, ref_socket)

    cam.focal = (intr[0][0], intr[1][1])
    cam.principal = (intr[0][2], intr[1][2])
    cam.size = resolution
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, dist)
    cam.rig_from_camera = _oak_extrinsics_to_pose(extr)
    cam.border_top = 50
    cam.border_bottom = 0
    cam.border_left = 70
    cam.border_right = 70
    return cam


def _plot(trajectory: List[np.ndarray]) -> None:
    if not trajectory:
        print("No trajectory to plot.")
        return

    pts = np.array(trajectory)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(pts[:, 0], pts[:, 2], linewidth=1.5, color="steelblue")  # X/Z = top-down
    ax.scatter(pts[0, 0], pts[0, 2], color="green", s=60, zorder=5, label="start")
    ax.scatter(pts[-1, 0], pts[-1, 2], color="red", s=60, zorder=5, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"OAK cuVSLAM trajectory  ({len(pts)} frames)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True)
    plt.tight_layout()
    out = "oak_slam_trajectory.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


def main() -> None:
    trajectory: List[np.ndarray] = []
    running = [True]

    def _on_sigint(_sig, _frame):
        running[0] = False

    signal.signal(signal.SIGINT, _on_sigint)

    # Pass a serial number string to target a specific device, e.g.:
    # device = dai.Device("1944301071E8975A00")
    device = dai.Device()
    calib = device.readCalibration()

    cameras = [
        _build_camera(calib, dai.CameraBoardSocket.CAM_B, RESOLUTION, dai.CameraBoardSocket.CAM_A),
        _build_camera(calib, dai.CameraBoardSocket.CAM_C, RESOLUTION, dai.CameraBoardSocket.CAM_A),
    ]

    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=False,
        enable_observations_export=False,
        rectified_stereo_camera=False,
    )
    tracker = vslam.Tracker(vslam.Rig(cameras), cfg)

    pipeline = dai.Pipeline(device)
    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=FPS)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=FPS)

    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5 / FPS))

    left_cam.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8).link(sync.inputs["left"])
    right_cam.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8).link(sync.inputs["right"])
    queue = sync.out.createOutputQueue()

    pipeline.start()
    print(f"Running — press Ctrl+C to stop and plot. Warming up {WARMUP_FRAMES} frames...")

    frame_id = 0
    prev_ts: Optional[int] = None

    while pipeline.isRunning() and running[0]:
        group: dai.MessageGroup = queue.get()
        ts = int(group.getTimestamp().total_seconds() * 1e9)

        if prev_ts is not None and (ts - prev_ts) > 35_000_000:
            print(f"  [warn] frame gap {(ts - prev_ts) / 1e6:.1f} ms")

        frame_id += 1
        prev_ts = ts

        if frame_id <= WARMUP_FRAMES:
            if frame_id == WARMUP_FRAMES:
                print("Warmup done, tracking...")
            continue

        left = group["left"].getCvFrame()
        right = group["right"].getCvFrame()

        result, _ = tracker.track(ts, (left, right))
        pose = result.world_from_rig
        if pose is None:
            print(f"  [warn] tracking lost at frame {frame_id}")
            continue

        trajectory.append(np.array(pose.pose.translation))

    print(f"\nStopped. Frames tracked: {len(trajectory)}")
    _plot(trajectory)


if __name__ == "__main__":
    main()
