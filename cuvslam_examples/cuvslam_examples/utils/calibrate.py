#!/usr/bin/env python3
"""
Calibrate the extrinsic (R, t) between two RealSense cameras (D435 or D455)
using a checkerboard pattern visible to both cameras simultaneously.

The rig origin is placed at the **midpoint** of the front camera (halfway
between its left and right IR sensors).  All camera transforms in the
output YAML are expressed w.r.t. this midpoint.

Workflow:
  1. Connect two RealSense cameras.
  2. Hold a checkerboard so BOTH cameras can see it (place the board in
     the overlapping region between the two fields of view).
  3. Press SPACE to capture a pair, ESC to finish collecting.
  4. The script runs cv2.stereoCalibrate and prints R, t.
  5. Optionally writes a rig YAML compatible with run_multicamera*.py.

Usage:
  # Interactive capture (live cameras, two D455):
  python calibrate_stereo_extrinsics.py \
      --front-serial 242422303248 --other-serial 339522301389 \
      --front-type d455 --other-type d455 \
      --rows 6 --cols 9 --square-size 0.025

  # Mixed D455 + D435:
  python calibrate_stereo_extrinsics.py \
      --front-serial 242422303248 --other-serial 339522301389 \
      --front-type d455 --other-type d435 \
      --rows 6 --cols 9 --square-size 0.025

  # From previously saved image pairs:
  python calibrate_stereo_extrinsics.py \
      --image-dir ./calib_images --rows 6 --cols 9 --square-size 0.025 \
      --front-type d455 --other-type d455

Arguments:
  --front-serial / --other-serial
                           RealSense serial numbers for the front and
                           other cameras.
  --front-type / --other-type
                           Camera model: 'd455' (baseline 95 mm) or
                           'd435' (baseline 50 mm).  Default: 'd455'.
  --rows / --cols          Inner corner count of the checkerboard.
  --square-size            Physical size of one square in metres.
  --image-dir              Directory to save/load image pairs.
  --min-pairs              Minimum number of valid pairs before calibrating (default 15).
  --output-yaml            Path for the output rig YAML file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Camera-model baselines (left IR → right IR distance, in metres)
# ---------------------------------------------------------------------------
CAMERA_BASELINES = {
    "d455": 0.095,
    "d435": 0.050,
}

# ---------------------------------------------------------------------------
# Checkerboard detection
# ---------------------------------------------------------------------------

def detect_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Detect checkerboard corners in a grayscale image.

    Args:
        image: Grayscale (H, W) uint8 image.
        pattern_size: (cols, rows) inner corners of the checkerboard.

    Returns:
        (N, 1, 2) float32 corner array, or None if detection failed.
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(image, pattern_size, flags=flags)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    return corners


def make_object_points(
    pattern_size: Tuple[int, int],
    square_size: float,
) -> np.ndarray:
    """Create 3-D checkerboard object points on the Z=0 plane.

    Args:
        pattern_size: (cols, rows) inner corners.
        square_size: Physical side length of one square (metres).

    Returns:
        (N, 3) float32 array.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


# ---------------------------------------------------------------------------
# RealSense helpers
# ---------------------------------------------------------------------------

def open_realsense_pipeline(
    serial: str,
    resolution: Tuple[int, int] = (1280, 800),
    fps: int = 15,
):
    """Open one RealSense camera streaming left IR only.

    Returns (pipeline, profile).
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.infrared, 1,
        resolution[0], resolution[1],
        rs.format.y8, fps,
    )

    # Disable the projector so the checkerboard is not washed out.
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    profile = pipeline.start(config)
    return pipeline, profile


def get_factory_intrinsics(
    profile,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read factory-calibrated intrinsics from a RealSense pipeline profile.

    Args:
        profile: An active rs.pipeline_profile.

    Returns:
        (camera_matrix, dist_coeffs) where camera_matrix is 3×3 and
        dist_coeffs is (5,) — both as float64 numpy arrays.
    """
    import pyrealsense2 as rs  # noqa: F811

    stream = profile.get_stream(rs.stream.infrared, 1)
    intr = stream.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([
        [intr.fx, 0.0,     intr.ppx],
        [0.0,     intr.fy, intr.ppy],
        [0.0,     0.0,     1.0],
    ], dtype=np.float64)

    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
    return camera_matrix, dist_coeffs


def grab_frame(pipeline) -> np.ndarray:
    """Wait for a frameset and return the left-IR image as uint8."""
    import pyrealsense2 as rs  # noqa: F811

    frames = pipeline.wait_for_frames()
    ir_frame = frames.get_infrared_frame(1)
    return np.asanyarray(ir_frame.get_data())


# ---------------------------------------------------------------------------
# Image-pair collection (live cameras)
# ---------------------------------------------------------------------------

def save_intrinsics(
    path: str,
    mtx_front: np.ndarray,
    dist_front: np.ndarray,
    mtx_other: np.ndarray,
    dist_other: np.ndarray,
) -> None:
    """Save factory intrinsics to a YAML file for offline reuse."""
    data = {
        "front": {
            "camera_matrix": mtx_front.tolist(),
            "dist_coeffs": dist_front.tolist(),
        },
        "other": {
            "camera_matrix": mtx_other.tolist(),
            "dist_coeffs": dist_other.tolist(),
        },
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)
    print(f"  Factory intrinsics saved to: {path}")


def load_intrinsics(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load factory intrinsics from a previously saved YAML file.

    Returns:
        (camera_matrix_front, dist_coeffs_front,
         camera_matrix_other, dist_coeffs_other)
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    # Support old ("cam1"/"cam2"), legacy ("front"/"right"), and
    # current ("front"/"other") key names.
    if "other" in data:
        front_key, other_key = "front", "other"
    elif "front" in data:
        front_key, other_key = "front", "right"
    else:
        front_key, other_key = "cam1", "cam2"
    mtx_front = np.array(data[front_key]["camera_matrix"], dtype=np.float64)
    dist_front = np.array(data[front_key]["dist_coeffs"], dtype=np.float64)
    mtx_other = np.array(data[other_key]["camera_matrix"], dtype=np.float64)
    dist_other = np.array(data[other_key]["dist_coeffs"], dtype=np.float64)
    return mtx_front, dist_front, mtx_other, dist_other


def collect_pairs_live(
    serial_front: str,
    serial_other: str,
    pattern_size: Tuple[int, int],
    min_pairs: int,
    save_dir: Optional[str],
    front_side: str = "left",
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int],
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interactively collect checkerboard image pairs from two cameras.

    Returns:
        (corners_front, corners_other, image_size,
         camera_matrix_front, dist_coeffs_front,
         camera_matrix_other, dist_coeffs_other)
        Each corners list has one (N,1,2) array per accepted pair.
        Intrinsics are factory-calibrated values from the RealSense devices.
    """
    pipe_front, prof_front = open_realsense_pipeline(serial_front)
    pipe_other, prof_other = open_realsense_pipeline(serial_other)

    # Read factory-calibrated intrinsics from the devices.
    mtx_front, dist_front = get_factory_intrinsics(prof_front)
    mtx_other, dist_other = get_factory_intrinsics(prof_other)
    print(f"  Factory intrinsics FRONT: fx={mtx_front[0,0]:.2f} fy={mtx_front[1,1]:.2f} "
          f"cx={mtx_front[0,2]:.2f} cy={mtx_front[1,2]:.2f}")
    print(f"  Factory intrinsics OTHER: fx={mtx_other[0,0]:.2f} fy={mtx_other[1,1]:.2f} "
          f"cx={mtx_other[0,2]:.2f} cy={mtx_other[1,2]:.2f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    corners_front_all: List[np.ndarray] = []
    corners_other_all: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None
    pair_idx = 0

    print(
        f"\nHold the checkerboard so BOTH cameras see it.\n"
        f"  Place the board in the overlapping FOV region.\n"
        f"  SPACE  = capture pair\n"
        f"  ESC    = finish collecting (need >= {min_pairs} valid pairs)\n"
    )

    while True:
        img_front = grab_frame(pipe_front)
        img_other = grab_frame(pipe_other)

        if image_size is None:
            image_size = (img_front.shape[1], img_front.shape[0])

        # Preview with corner overlay
        vis_front = cv2.cvtColor(img_front, cv2.COLOR_GRAY2BGR)
        vis_other = cv2.cvtColor(img_other, cv2.COLOR_GRAY2BGR)
        c_front = detect_corners(img_front, pattern_size)
        c_other = detect_corners(img_other, pattern_size)
        if c_front is not None:
            cv2.drawChessboardCorners(vis_front, pattern_size, c_front, True)
        if c_other is not None:
            cv2.drawChessboardCorners(vis_other, pattern_size, c_other, True)

        if front_side == "left":
            combined = np.hstack([vis_front, vis_other])
            title = "Front | Other  [SPACE=capture, ESC=done]"
        else:
            combined = np.hstack([vis_other, vis_front])
            title = "Other | Front  [SPACE=capture, ESC=done]"
        combined = cv2.resize(combined, None, fx=0.5, fy=0.5,
                              interpolation=cv2.INTER_AREA)
        status = f"Valid pairs: {len(corners_front_all)}"
        cv2.putText(
            combined, status, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.imshow(title, combined)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            break
        if key == ord(" "):
            if c_front is not None and c_other is not None:
                corners_front_all.append(c_front)
                corners_other_all.append(c_other)
                print(f"  Pair {pair_idx} captured  ✓")
                if save_dir:
                    cv2.imwrite(
                        os.path.join(save_dir, f"cam1_{pair_idx:03d}.png"),
                        img_front,
                    )
                    cv2.imwrite(
                        os.path.join(save_dir, f"cam2_{pair_idx:03d}.png"),
                        img_other,
                    )
                pair_idx += 1
            else:
                print("  Checkerboard NOT detected in both views — skipped")

    cv2.destroyAllWindows()
    pipe_front.stop()
    pipe_other.stop()

    if image_size is None:
        sys.exit("No frames received.")

    # Save factory intrinsics alongside images for offline reuse.
    if save_dir:
        intrinsics_path = os.path.join(save_dir, "factory_intrinsics.yaml")
        save_intrinsics(intrinsics_path, mtx_front, dist_front,
                        mtx_other, dist_other)

    return (corners_front_all, corners_other_all, image_size,
            mtx_front, dist_front, mtx_other, dist_other)


# ---------------------------------------------------------------------------
# Image-pair collection (from disk)
# ---------------------------------------------------------------------------

def collect_pairs_from_disk(
    image_dir: str,
    pattern_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    """Load previously saved cam1_*.png (front) / cam2_*.png (other) pairs.

    Returns:
        (corners_front, corners_other, image_size)
    """
    cam1_files = sorted(Path(image_dir).glob("cam1_*.png"))
    cam2_files = sorted(Path(image_dir).glob("cam2_*.png"))
    if len(cam1_files) != len(cam2_files):
        sys.exit(
            f"Mismatch: {len(cam1_files)} front-camera images vs "
            f"{len(cam2_files)} other-camera images."
        )

    corners_front_all: List[np.ndarray] = []
    corners_other_all: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    for f1, f2 in zip(cam1_files, cam2_files):
        img1 = cv2.imread(str(f1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(f2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue
        if image_size is None:
            image_size = (img1.shape[1], img1.shape[0])

        c_front = detect_corners(img1, pattern_size)
        c_other = detect_corners(img2, pattern_size)
        if c_front is not None and c_other is not None:
            corners_front_all.append(c_front)
            corners_other_all.append(c_other)
            print(f"  {f1.name} (front) + {f2.name} (other)  ✓")
        else:
            print(f"  {f1.name} + {f2.name}  ✗ (corners not found)")

    if image_size is None:
        sys.exit("No valid images found.")
    return corners_front_all, corners_other_all, image_size


# ---------------------------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------------------------

def run_stereo_calibration(
    corners_front: List[np.ndarray],
    corners_other: List[np.ndarray],
    image_size: Tuple[int, int],
    pattern_size: Tuple[int, int],
    square_size: float,
    factory_intrinsics: Optional[Tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run OpenCV stereo calibration and return extrinsics.

    Args:
        corners_front: Corner arrays from the front camera (left IR).
        corners_other: Corner arrays from the other camera (left IR).
        image_size: (width, height) of the images.
        pattern_size: (cols, rows) inner corners.
        square_size: Physical square side in metres.
        factory_intrinsics: Optional (mtx_front, dist_front,
            mtx_other, dist_other) from the RealSense factory calibration.
            When provided, intrinsics are fixed and only the extrinsic
            R, t is estimated.

    Returns:
        (R, t, camera_matrix_front, camera_matrix_other, rms_reprojection_error)
        R is 3×3 rotation, t is (3,1) translation (front-left-IR → other-left-IR).
    """
    objp = make_object_points(pattern_size, square_size)
    object_points = [objp] * len(corners_front)

    if factory_intrinsics is not None:
        # Use factory-calibrated intrinsics — skip unreliable mono estimation.
        mtx_f, dist_f, mtx_s, dist_s = factory_intrinsics
        print("  Using factory intrinsics (fixed).")
    else:
        # Fall back to mono calibration for initial intrinsics.
        print("  Estimating intrinsics from checkerboard (no factory data).")
        flags_mono = cv2.CALIB_FIX_K3
        _, mtx_f, dist_f, _, _ = cv2.calibrateCamera(
            object_points, corners_front, image_size, None, None,
            flags=flags_mono,
        )
        _, mtx_s, dist_s, _, _ = cv2.calibrateCamera(
            object_points, corners_other, image_size, None, None,
            flags=flags_mono,
        )

    # Stereo calibration — fix intrinsics, solve only for R, t.
    flags_stereo = cv2.CALIB_FIX_INTRINSIC
    rms, mtx_f, dist_f, mtx_s, dist_s, R, t, _, _ = cv2.stereoCalibrate(
        object_points,
        corners_front,
        corners_other,
        mtx_f, dist_f,
        mtx_s, dist_s,
        image_size,
        flags=flags_stereo,
        criteria=(
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6
        ),
    )
    return R, t, mtx_f, mtx_s, rms


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_results(
    R: np.ndarray,
    t: np.ndarray,
    mtx_front: np.ndarray,
    mtx_other: np.ndarray,
    rms: float,
    num_pairs: int,
) -> None:
    """Pretty-print calibration results to console."""
    print("\n" + "=" * 60)
    print("  Stereo Extrinsic Calibration Results  (front + other)")
    print("=" * 60)
    print(f"  Valid pairs used:    {num_pairs}")
    print(f"  RMS reprojection:    {rms:.4f} px")
    print()
    print("  Rotation matrix (front-left-IR → other-left-IR):")
    for row in R:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
    print()
    print(f"  Translation (front-left-IR → other-left-IR) [metres]:")
    print(f"    tx = {t[0, 0]:+.6f}")
    print(f"    ty = {t[1, 0]:+.6f}")
    print(f"    tz = {t[2, 0]:+.6f}")
    print(f"    |t| = {np.linalg.norm(t):.6f}")
    print()
    print("  Front camera intrinsics:")
    print(f"    fx={mtx_front[0,0]:.2f}  fy={mtx_front[1,1]:.2f}  "
          f"cx={mtx_front[0,2]:.2f}  cy={mtx_front[1,2]:.2f}")
    print("  Other camera intrinsics:")
    print(f"    fx={mtx_other[0,0]:.2f}  fy={mtx_other[1,1]:.2f}  "
          f"cx={mtx_other[0,2]:.2f}  cy={mtx_other[1,2]:.2f}")
    print("=" * 60)


def write_rig_yaml(
    path: str,
    serial_front: str,
    serial_other: str,
    R: np.ndarray,
    t: np.ndarray,
    baseline_front: float,
    baseline_other: float,
) -> None:
    """Write a multi-camera rig YAML compatible with run_multicamera*.py.

    The rig origin is placed at the **midpoint** of the front camera
    (halfway between its left and right IR sensors).  All camera
    ``rig_from_camera`` transforms are expressed w.r.t. this midpoint.

    OpenCV ``stereoCalibrate`` gives ``(R, t)`` that maps points from the
    front-left-IR frame into the other-left-IR frame.  We re-express
    everything relative to the front camera midpoint by shifting all
    translations by ``[-baseline_front / 2, 0, 0]``.

    Args:
        path: Output file path.
        serial_front: Serial number of the front camera.
        serial_other: Serial number of the other camera.
        R: 3×3 rotation (front-left-IR → other-left-IR from stereoCalibrate).
        t: (3,1) translation (front-left-IR → other-left-IR, metres).
        baseline_front: Intra-camera stereo baseline of the front camera
            (left→right IR) in metres.
        baseline_other: Intra-camera stereo baseline of the other camera
            (left→right IR) in metres.
    """
    half_bf = baseline_front / 2.0
    offset = np.array([half_bf, 0.0, 0.0])  # shift from front-left-IR to midpoint

    def mat_row(r: np.ndarray, tx: float) -> list:
        return [round(float(r[0]), 6), round(float(r[1]), 6),
                round(float(r[2]), 6), round(float(tx), 6)]

    # --- Front camera (identity rotation, centred on midpoint) -----------
    # left IR  at  [-half_bf, 0, 0]  relative to midpoint
    # right IR at  [+half_bf, 0, 0]  relative to midpoint
    front_left = [
        [1, 0, 0, round(-half_bf, 6)],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.0],
    ]
    front_right = [
        [1, 0, 0, round(+half_bf, 6)],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.0],
    ]

    # --- Other camera (rotated, translated w.r.t. midpoint) ---------------
    # stereoCalibrate outputs in OpenCV convention (x-right, y-down, z-forward).
    # Rig YAML uses x-right, y-up, z-back.  Convert with C = diag(1, -1, -1).
    # Invert stereoCalibrate (p_other = R*p_front + t) to get rig_from_other,
    # then apply convention flip: R_out = C @ R^T @ C,  t_out = C @ (-R^T @ t).
    C = np.diag([1.0, -1.0, -1.0])
    Rt = R.T
    t_flat = t.flatten()
    R_out = C @ Rt @ C
    t_other_left = C @ (-Rt @ t_flat) - offset
    other_left = [
        mat_row(R_out[0], t_other_left[0]),
        mat_row(R_out[1], t_other_left[1]),
        mat_row(R_out[2], t_other_left[2]),
    ]
    # Other camera's right IR is at [baseline, 0, 0] in its local frame.
    t_other_right = C @ (Rt @ np.array([baseline_other, 0.0, 0.0]) - Rt @ t_flat) - offset
    other_right = [
        mat_row(R_out[0], t_other_right[0]),
        mat_row(R_out[1], t_other_right[1]),
        mat_row(R_out[2], t_other_right[2]),
    ]

    rig = {
        "stereo_cameras": [
            {
                "serial": serial_front,
                "left_camera": {"transform": front_left},
                "right_camera": {"transform": front_right},
            },
            {
                "serial": serial_other,
                "left_camera": {"transform": other_left},
                "right_camera": {"transform": other_right},
            },
        ]
    }

    with open(path, "w") as f:
        yaml.dump(rig, f, default_flow_style=None, sort_keys=False)
    print(f"\nRig YAML written to: {path}")
    print(f"  Rig origin = front camera midpoint")
    print(f"  Front camera baseline = {baseline_front:.4f} m")
    print(f"  Other camera baseline = {baseline_other:.4f} m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stereo extrinsic calibration via checkerboard "
            "(two RealSense cameras, rig origin at front camera midpoint)."
        ),
    )
    parser.add_argument("--front-serial", type=str, default=None,
                        help="RealSense serial number for the FRONT camera.")
    parser.add_argument("--other-serial", type=str, default=None,
                        help="RealSense serial number for the OTHER camera.")
    parser.add_argument("--front-type", type=str, default="d455",
                        choices=["d455", "d435"],
                        help="Model of the front camera: 'd455' (baseline "
                             "95 mm) or 'd435' (baseline 50 mm). Default: d455.")
    parser.add_argument("--other-type", type=str, default="d455",
                        choices=["d455", "d435"],
                        help="Model of the other camera: 'd455' (baseline "
                             "95 mm) or 'd435' (baseline 50 mm). Default: d455.")
    parser.add_argument("--rows", type=int, default=6,
                        help="Checkerboard inner corner rows.")
    parser.add_argument("--cols", type=int, default=9,
                        help="Checkerboard inner corner columns.")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="Physical square size in metres (default 25 mm).")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory to save/load image pairs.")
    parser.add_argument("--min-pairs", type=int, default=15,
                        help="Minimum valid pairs before calibrating.")
    parser.add_argument("--front-side", type=str, default="left",
                        choices=["left", "right"],
                        help="Which side to display the front camera in the "
                             "preview window. Default: left.")
    parser.add_argument("--output-yaml", type=str, default=None,
                        help="Path to write rig YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern_size = (args.cols, args.rows)

    # Derive intra-camera baselines from camera type.
    baseline_front = CAMERA_BASELINES[args.front_type]
    baseline_other = CAMERA_BASELINES[args.other_type]
    print(f"  Front camera type: {args.front_type.upper()}  "
          f"(baseline {baseline_front * 1000:.0f} mm)")
    print(f"  Other camera type: {args.other_type.upper()}  "
          f"(baseline {baseline_other * 1000:.0f} mm)")

    # Decide capture mode: live cameras or pre-saved images.
    have_serials = (args.front_serial is not None
                    and args.other_serial is not None)
    have_images = args.image_dir is not None and os.path.isdir(args.image_dir)
    factory_intrinsics = None

    if have_serials:
        (corners_front, corners_other, image_size,
         mtx_f, d_f, mtx_r, d_r) = collect_pairs_live(
            args.front_serial, args.other_serial, pattern_size,
            args.min_pairs, args.image_dir, args.front_side,
        )
        factory_intrinsics = (mtx_f, d_f, mtx_r, d_r)
    elif have_images:
        corners_front, corners_other, image_size = collect_pairs_from_disk(
            args.image_dir, pattern_size,
        )
        # Try to load factory intrinsics saved during live capture.
        intrinsics_path = os.path.join(args.image_dir,
                                       "factory_intrinsics.yaml")
        if os.path.isfile(intrinsics_path):
            factory_intrinsics = load_intrinsics(intrinsics_path)
            print(f"  Loaded factory intrinsics from: {intrinsics_path}")
        else:
            print("  No factory_intrinsics.yaml found — "
                  "will estimate intrinsics from checkerboard.")
    else:
        sys.exit(
            "Provide --front-serial and --other-serial for live capture, "
            "or --image-dir with existing images."
        )

    if len(corners_front) < args.min_pairs:
        sys.exit(
            f"Only {len(corners_front)} valid pairs — "
            f"need at least {args.min_pairs}. Capture more."
        )

    print(f"\nRunning stereo calibration with {len(corners_front)} pairs ...")
    R, t, mtx_front, mtx_other, rms = run_stereo_calibration(
        corners_front, corners_other, image_size, pattern_size,
        args.square_size, factory_intrinsics=factory_intrinsics,
    )
    print_results(R, t, mtx_front, mtx_other, rms, len(corners_front))

    if args.output_yaml and have_serials:
        write_rig_yaml(
            args.output_yaml,
            args.front_serial,
            args.other_serial,
            R, t,
            baseline_front=baseline_front,
            baseline_other=baseline_other,
        )


if __name__ == "__main__":
    main()
