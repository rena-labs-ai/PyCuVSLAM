"""Pure trajectory-comparison metrics and plotting (no ROS).

Used by the optional odom-logger node to render a reference-vs-estimated
trajectory PNG with ATE. Unit-testable in isolation.
"""

import math
from pathlib import Path
from typing import List, Tuple


def compute_ate(
    ref: List[Tuple[float, float]], est: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """ATE RMSE and mean over paired 2D poses."""
    n = min(len(ref), len(est))
    if n == 0:
        return 0.0, 0.0
    errors = [math.hypot(r[0] - e[0], r[1] - e[1]) for r, e in zip(ref[:n], est[:n])]
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mean = sum(errors) / n
    return rmse, mean


# Color palette for estimated trajectories (reference is always blue).
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
    ref: List[tuple],
    estimated_list: List[Tuple[str, List[tuple]]],
    title: str,
    out_path: Path,
) -> dict:
    """Plot one reference trajectory against multiple estimated trajectories.

    Layout: left = 2D XY trajectories, right = ATE over time. Returns
    {label: ate_rmse} for each estimated trajectory that had a paired ref.
    matplotlib is imported lazily so importing this module is cheap.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_traj, ax_ate) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]}
    )

    if ref:
        xs, ys = [p[0] for p in ref], [p[1] for p in ref]
        ax_traj.plot(xs, ys, "-", label="reference", linewidth=2, color=_REF_COLOR)
        ax_traj.plot(xs[0], ys[0], "o", color=_REF_COLOR, markersize=8)
        ax_traj.plot(xs[-1], ys[-1], "s", color=_REF_COLOR, markersize=8)

    ate_results: dict = {}
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
            ax_ate.plot(range(n), ate_errors, "-", linewidth=1.2, color=color, label=label)
            ax_ate.fill_between(range(n), ate_errors, alpha=0.10, color=color)

    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")

    ax_ate.set_title("ATE over time")
    ax_ate.set_xlabel("sample")
    ax_ate.set_ylabel("error (m)")
    ax_ate.grid(True, alpha=0.3)

    fig.text(0.5, 0.02, title, ha="center", va="bottom", fontweight="bold", fontsize=11)

    handles, labels = ax_traj.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.08),
            ncol=len(handles), fontsize=8, framealpha=0.9,
        )

    if ate_results:
        summary = "\n".join(
            f"{lbl:35s}  ATE RMSE = {v:.4f} m" for lbl, v in sorted(ate_results.items())
        )
        fig.text(
            0.99, 0.02, summary, fontsize=9, family="monospace", va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.6),
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return ate_results
