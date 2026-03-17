#!/usr/bin/env python3
"""Read odom_experiments.log, extract an experiment by name, plot ground_truth vs other trajectories."""

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_odom_log(log_path: Path, experiment_name: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    content = log_path.read_text()
    pattern = rf"{re.escape(experiment_name)}\s*\n\*\*\*\*\*\s*\n(.*?)(?=\n+[a-z_][a-z0-9_]*\s*\n\*\*\*\*\*|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Experiment '{experiment_name}' not found in {log_path}")

    block = match.group(1)
    ground_truth = []
    other = []
    odom_re = re.compile(r"\[(\d+)\]\s+(ground_truth|other):\s+([-\d.]+),([-\d.]+),([-\d.]+)")

    for m in odom_re.finditer(block):
        _, source, x, y, z = m.groups()
        x, y = float(x), float(y)
        if source == "ground_truth":
            ground_truth.append((x, y))
        else:
            other.append((x, y))

    return ground_truth, other


def compute_ate(ref: list[tuple[float, float]], est: list[tuple[float, float]]) -> tuple[float, float]:
    """ATE RMSE and mean over paired 2D poses."""
    n = min(len(ref), len(est))
    if n == 0:
        return 0.0, 0.0
    errors = [math.hypot(r[0] - e[0], r[1] - e[1]) for r, e in zip(ref[:n], est[:n])]
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mean = sum(errors) / n
    return rmse, mean


def compute_rpe(ref: list[tuple[float, float]], est: list[tuple[float, float]], delta: int = 1) -> tuple[float, float]:
    """RPE (translational) RMSE and mean over paired 2D poses with given step delta."""
    n = min(len(ref), len(est))
    if n <= delta:
        return 0.0, 0.0
    errors = []
    for i in range(n - delta):
        gt_dx = ref[i + delta][0] - ref[i][0]
        gt_dy = ref[i + delta][1] - ref[i][1]
        est_dx = est[i + delta][0] - est[i][0]
        est_dy = est[i + delta][1] - est[i][1]
        errors.append(math.hypot(gt_dx - est_dx, gt_dy - est_dy))
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mean = sum(errors) / len(errors)
    return rmse, mean


def _normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def compute_rpe_rotation(
    ref: list[tuple[float, float, float]],
    est: list[tuple[float, float, float]],
    delta: int = 1,
) -> tuple[float, float]:
    """RPE (rotational) RMSE and mean in radians. Tuples must be (x, y, yaw)."""
    n = min(len(ref), len(est))
    if n <= delta:
        return 0.0, 0.0
    errors = []
    for i in range(n - delta):
        gt_dyaw = _normalize_angle(ref[i + delta][2] - ref[i][2])
        est_dyaw = _normalize_angle(est[i + delta][2] - est[i][2])
        errors.append(abs(_normalize_angle(gt_dyaw - est_dyaw)))
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mean = sum(errors) / len(errors)
    return rmse, mean


def _per_sample_rpe_translation(ref, est, delta=1) -> list[float]:
    n = min(len(ref), len(est))
    if n <= delta:
        return []
    return [
        math.hypot(
            (ref[i + delta][0] - ref[i][0]) - (est[i + delta][0] - est[i][0]),
            (ref[i + delta][1] - ref[i][1]) - (est[i + delta][1] - est[i][1]),
        )
        for i in range(n - delta)
    ]


def _per_sample_rpe_rotation(ref, est, delta=1) -> list[float]:
    n = min(len(ref), len(est))
    if n <= delta:
        return []
    errors = []
    for i in range(n - delta):
        gt_dyaw = _normalize_angle(ref[i + delta][2] - ref[i][2])
        est_dyaw = _normalize_angle(est[i + delta][2] - est[i][2])
        errors.append(abs(_normalize_angle(gt_dyaw - est_dyaw)))
    return errors


def plot(
    ground_truth: list[tuple[float, float]],
    other: list[tuple[float, float]],
    experiment_name: str,
    out_path: Path,
    metrics: dict[str, float] | None = None,
    jitter_points: list[tuple[float, float]] | None = None,
) -> None:
    if not ground_truth and not other:
        raise ValueError("No trajectory data")

    if metrics is None:
        ate_rmse, ate_mean = compute_ate(ground_truth, other)
        rpe_rmse, rpe_mean = compute_rpe(ground_truth, other)
        rpe_rot_rmse, rpe_rot_mean = 0.0, 0.0
    else:
        ate_rmse = metrics["ate_rmse"]
        ate_mean = metrics["ate_mean"]
        rpe_rmse = metrics["rpe_rmse"]
        rpe_mean = metrics["rpe_mean"]
        rpe_rot_rmse = metrics.get("rpe_rot_rmse", 0.0)
        rpe_rot_mean = metrics.get("rpe_rot_mean", 0.0)

    n_samples = min(len(ground_truth), len(other))
    ate_errors = [
        math.hypot(ground_truth[i][0] - other[i][0], ground_truth[i][1] - other[i][1])
        for i in range(n_samples)
    ]
    rpe_trans_errors = _per_sample_rpe_translation(ground_truth, other)
    has_yaw = n_samples > 0 and len(ground_truth[0]) >= 3 and len(other[0]) >= 3
    rpe_rot_errors = _per_sample_rpe_rotation(ground_truth, other) if has_yaw else []

    gs = plt.GridSpec(3, 2, width_ratios=[3, 1.5], hspace=0.45)
    fig = plt.figure(figsize=(16, 10))

    ax_traj = fig.add_subplot(gs[:, 0])
    if ground_truth:
        xf, yf = [p[0] for p in ground_truth], [p[1] for p in ground_truth]
        ax_traj.plot(xf, yf, "-", label="ground truth", linewidth=2, color="#1f77b4")
        ax_traj.plot(xf[0], yf[0], "o", color="#1f77b4", markersize=8)
        ax_traj.plot(xf[-1], yf[-1], "s", color="#1f77b4", markersize=8)
    if other:
        xc, yc = [p[0] for p in other], [p[1] for p in other]
        ax_traj.plot(xc, yc, "--", label="estimated", linewidth=2, color="#ff7f0e")
        ax_traj.plot(xc[0], yc[0], "o", color="#ff7f0e", markersize=8)
        ax_traj.plot(xc[-1], yc[-1], "s", color="#ff7f0e", markersize=8)
    if jitter_points:
        jx = [p[0] for p in jitter_points]
        jy = [p[1] for p in jitter_points]
        ax_traj.scatter(jx, jy, s=12, c="red", zorder=5, label=f"jitter ({len(jitter_points)})")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="upper left")
    ax_traj.set_title(experiment_name, fontweight="bold")
    ax_traj.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"x: {v:.2f}"))
    ax_traj.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"y: {v:.2f}"))
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")

    ax_ate = fig.add_subplot(gs[0, 1])
    if ate_errors:
        ax_ate.plot(range(n_samples), ate_errors, "-", linewidth=1.2, color="#d62728")
        ax_ate.axhline(ate_mean, ls="--", color="#888", lw=1, label=f"mean={ate_mean:.4f}m")
        ax_ate.fill_between(range(n_samples), ate_errors, alpha=0.15, color="#d62728")
    ax_ate.set_ylabel("error (m)")
    ax_ate.set_title("ATE over time")
    ax_ate.grid(True, alpha=0.3)
    ax_ate.legend(loc="upper left", fontsize=7)

    ax_rpe_t = fig.add_subplot(gs[1, 1])
    if rpe_trans_errors:
        ax_rpe_t.plot(range(len(rpe_trans_errors)), rpe_trans_errors, "-", linewidth=1.2, color="#2ca02c")
        ax_rpe_t.axhline(rpe_mean, ls="--", color="#888", lw=1, label=f"mean={rpe_mean:.4f}m")
        ax_rpe_t.fill_between(range(len(rpe_trans_errors)), rpe_trans_errors, alpha=0.15, color="#2ca02c")
    ax_rpe_t.set_ylabel("error (m)")
    ax_rpe_t.set_title("RPE translation over time")
    ax_rpe_t.grid(True, alpha=0.3)
    ax_rpe_t.legend(loc="upper left", fontsize=7)

    ax_rpe_r = fig.add_subplot(gs[2, 1])
    if rpe_rot_errors:
        rpe_rot_deg = [math.degrees(e) for e in rpe_rot_errors]
        ax_rpe_r.plot(range(len(rpe_rot_deg)), rpe_rot_deg, "-", linewidth=1.2, color="#9467bd")
        ax_rpe_r.axhline(math.degrees(rpe_rot_mean), ls="--", color="#888", lw=1,
                         label=f"mean={math.degrees(rpe_rot_mean):.4f}°")
        ax_rpe_r.fill_between(range(len(rpe_rot_deg)), rpe_rot_deg, alpha=0.15, color="#9467bd")
    ax_rpe_r.set_xlabel("sample")
    ax_rpe_r.set_ylabel("error (deg)")
    ax_rpe_r.set_title("RPE rotation over time")
    ax_rpe_r.grid(True, alpha=0.3)
    ax_rpe_r.legend(loc="upper left", fontsize=7)

    metrics_text = (
        f"ATE RMSE:     {ate_rmse:.4f} m\n"
        f"ATE Mean:     {ate_mean:.4f} m\n"
        f"RPE RMSE:     {rpe_rmse:.4f} m\n"
        f"RPE Mean:     {rpe_mean:.4f} m\n"
        f"RPE Rot RMSE: {math.degrees(rpe_rot_rmse):.4f} deg\n"
        f"RPE Rot Mean: {math.degrees(rpe_rot_mean):.4f} deg\n"
        f"Samples:      {n_samples}"
    )
    fig.text(
        0.99, 0.01, metrics_text, fontsize=9, family="monospace",
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.6),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot odom trajectories for an experiment")
    parser.add_argument("experiment", help="Experiment name (e.g. stereo_air, stereo_1)")
    parser.add_argument("--log", type=Path, default=Path("odom_experiments.log"), help="Path to odom log file")
    parser.add_argument("-o", "--output", type=Path, help="Output file (default: experiments/<experiment>.png)")
    args = parser.parse_args()

    ground_truth, other = parse_odom_log(args.log, args.experiment)
    out = args.output or Path(f"experiments/{args.experiment}.png")
    out.parent.mkdir(exist_ok=True)

    plot(ground_truth, other, args.experiment, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
