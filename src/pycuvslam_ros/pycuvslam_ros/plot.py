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


# ---------------------------------------------------------------------------
# Color palette for estimated trajectories (reference is always blue)
# ---------------------------------------------------------------------------
_REF_COLOR = "#1f77b4"
_EST_COLORS = [
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # teal
]


def plot_combined(
    ref: list[tuple],
    estimated_list: list[tuple[str, list[tuple]]],
    title: str,
    out_path: Path,
) -> dict[str, float]:
    """Plot one reference trajectory against multiple estimated trajectories.

    Layout: left = 2D XY trajectories, right = ATE over time (all on one panel).
    Returns {label: ate_rmse} for each estimated trajectory that had a paired ref.
    """
    fig, (ax_traj, ax_ate) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]}
    )

    if ref:
        xs, ys = [p[0] for p in ref], [p[1] for p in ref]
        ax_traj.plot(xs, ys, "-", label="reference", linewidth=2, color=_REF_COLOR)
        ax_traj.plot(xs[0], ys[0], "o", color=_REF_COLOR, markersize=8)
        ax_traj.plot(xs[-1], ys[-1], "s", color=_REF_COLOR, markersize=8)

    ate_results: dict[str, float] = {}
    for i, (label, est) in enumerate(estimated_list):
        color = _EST_COLORS[i % len(_EST_COLORS)]
        if est:
            xs, ys = [p[0] for p in est], [p[1] for p in est]
            ax_traj.plot(xs, ys, "--", label=label, linewidth=1.5, color=color)
            ax_traj.plot(xs[0], ys[0], "o", color=color, markersize=6)
            ax_traj.plot(xs[-1], ys[-1], "s", color=color, markersize=6)
        if ref and est:
            n = min(len(ref), len(est))
            ate_rmse, _ = compute_ate(ref[:n], est[:n])
            ate_results[label] = ate_rmse
            ate_errors = [
                math.hypot(ref[j][0] - est[j][0], ref[j][1] - est[j][1])
                for j in range(n)
            ]
            ax_ate.plot(
                range(n), ate_errors, "-", linewidth=1.2, color=color,
                label=f"{label}  RMSE={ate_rmse:.3f}m",
            )
            ax_ate.fill_between(range(n), ate_errors, alpha=0.10, color=color)

    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="upper left", fontsize=8)
    ax_traj.set_title(title, fontweight="bold")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")

    ax_ate.set_title("ATE over time")
    ax_ate.set_xlabel("sample")
    ax_ate.set_ylabel("error (m)")
    ax_ate.grid(True, alpha=0.3)
    ax_ate.legend(loc="upper left", fontsize=7)

    n_summary_lines = len(ate_results)
    bottom_margin = 0.04 + n_summary_lines * 0.045

    if ate_results:
        summary = "\n".join(
            f"{lbl:35s}  ATE RMSE = {v:.4f} m"
            for lbl, v in sorted(ate_results.items())
        )
        fig.text(
            0.99, 0.01, summary, fontsize=9, family="monospace",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.6),
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_margin)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return ate_results


def plot(
    ground_truth: list[tuple[float, float]],
    other: list[tuple[float, float]],
    experiment_name: str,
    out_path: Path,
    metrics: dict[str, float] | None = None,
    jitter_points: list[tuple[float, float]] | None = None,
) -> None:
    """Single-experiment live preview plot (reference vs one estimated trajectory)."""
    if not ground_truth and not other:
        raise ValueError("No trajectory data")

    ate_rmse, ate_mean = (
        (metrics["ate_rmse"], metrics["ate_mean"]) if metrics
        else compute_ate(ground_truth, other)
    )
    n_samples = min(len(ground_truth), len(other))
    ate_errors = [
        math.hypot(ground_truth[i][0] - other[i][0], ground_truth[i][1] - other[i][1])
        for i in range(n_samples)
    ]

    fig, (ax_traj, ax_ate) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    if ground_truth:
        xf, yf = [p[0] for p in ground_truth], [p[1] for p in ground_truth]
        ax_traj.plot(xf, yf, "-", label="reference", linewidth=2, color=_REF_COLOR)
        ax_traj.plot(xf[0], yf[0], "o", color=_REF_COLOR, markersize=8)
        ax_traj.plot(xf[-1], yf[-1], "s", color=_REF_COLOR, markersize=8)
    if other:
        xc, yc = [p[0] for p in other], [p[1] for p in other]
        ax_traj.plot(xc, yc, "--", label="estimated", linewidth=2, color=_EST_COLORS[0])
        ax_traj.plot(xc[0], yc[0], "o", color=_EST_COLORS[0], markersize=8)
        ax_traj.plot(xc[-1], yc[-1], "s", color=_EST_COLORS[0], markersize=8)
    if jitter_points:
        ax_traj.scatter(
            [p[0] for p in jitter_points], [p[1] for p in jitter_points],
            s=12, c="red", zorder=5, label=f"jitter ({len(jitter_points)})",
        )
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="upper left")
    ax_traj.set_title(experiment_name, fontweight="bold")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")

    if ate_errors:
        ax_ate.plot(range(n_samples), ate_errors, "-", linewidth=1.2, color="#d62728")
        ax_ate.axhline(ate_mean, ls="--", color="#888", lw=1, label=f"mean={ate_mean:.4f}m")
        ax_ate.fill_between(range(n_samples), ate_errors, alpha=0.15, color="#d62728")
    ax_ate.set_title("ATE over time")
    ax_ate.set_xlabel("sample")
    ax_ate.set_ylabel("error (m)")
    ax_ate.grid(True, alpha=0.3)
    ax_ate.legend(loc="upper left", fontsize=7)

    fig.text(
        0.99, 0.01,
        f"ATE RMSE: {ate_rmse:.4f} m\nSamples:  {n_samples}",
        fontsize=9, family="monospace", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.6),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
