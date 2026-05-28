"""
Shared plotting functions for run_experiment.py.

plot_training_curves():
  - One figure, all models.
  - X-axis: episode number.  Y-axis: smoothed cumulative episode reward.

plot_test_results():
  - 3-subplot figure for one seed's test results across all models.
    1. Average AR per model (y max 1.0).
    2. Success rate vs failure rate per model (stacked bar).
    3. Cap-violation rate and conflict-violation rate per model (side-by-side bar).
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


MODEL_COLORS = {
    "P3_PPO":       "#4c78a8",
    "P4_MaskPPO":   "#f58518",
    "P5_LagPPO":    "#e45756",
    "P6_RepairPPO": "#72b7b2",
    "DQN":          "#54a24b",
    "DDQN":         "#b279a2",
    "P7_SeqPPO":    "#e377c2",
}

_DEFAULT_COLOR = "#aaaaaa"


def _moving_avg(vals: list[float], w: int) -> tuple[np.ndarray, int]:
    if len(vals) < w:
        return np.array(vals, dtype=float), 0
    kernel = np.ones(w) / w
    smoothed = np.convolve(vals, kernel, mode="valid")
    return smoothed, (w - 1) // 2


def _plot_curves_to_ax(ax, training_data: dict[str, dict], smooth_w: int, title: str) -> None:
    for model_name, data in training_data.items():
        ep_nums = np.array(data["episode_nums"],    dtype=float)
        ep_rews = np.array(data["episode_rewards"], dtype=float)
        color   = MODEL_COLORS.get(model_name, _DEFAULT_COLOR)

        ax.plot(ep_nums, ep_rews, color=color, alpha=0.15, linewidth=0.6)

        sm, off = _moving_avg(ep_rews, smooth_w)
        if len(sm) > 0:
            ax.plot(ep_nums[off: off + len(sm)], sm,
                    color=color, linewidth=2.0, label=model_name)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)


def plot_training_curves(
    training_data: dict[str, dict],
    outdir: Path,
    smooth_w: int = 2000,
    seed: int | None = None,
) -> None:
    """
    training_data: {model_name: {"episode_nums": [...], "episode_rewards": [...]}}
    P3_PPO is excluded from the combined plot and saved separately at outdir/p3/training_curve.png.
    """
    title_suffix = f"  (seed={seed})" if seed is not None else ""

    p3_data   = {k: v for k, v in training_data.items() if k == "P3_PPO"}
    main_data = {k: v for k, v in training_data.items() if k != "P3_PPO"}

    # Combined plot (no P3)
    if main_data:
        fig, ax = plt.subplots(figsize=(12, 5))
        _plot_curves_to_ax(ax, main_data, smooth_w, f"Training Reward vs Episode{title_suffix}")
        plt.tight_layout()
        path = outdir / "training_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")

    # P3-only plot — saved at seeds/{n}/p3/training_curve.png
    if p3_data:
        p3_dir = outdir.parent.parent / "p3"
        p3_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        _plot_curves_to_ax(ax, p3_data, smooth_w,
                           f"P3_PPO Training Reward vs Episode{title_suffix}")
        plt.tight_layout()
        path = p3_dir / "training_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")


def plot_test_results(
    agg_results: dict[str, dict],
    outdir: Path,
    seed: int | None = None,
) -> None:
    """
    agg_results: {model_name: aggregate_eval(...) dict}
    Produces a 3-subplot landscape figure (AR | Success/Fail | Violations).
    """
    model_names = [m for m in agg_results.keys() if m != "P3_PPO"]
    n = len(model_names)
    x = np.arange(n)
    colors = [MODEL_COLORS.get(m, _DEFAULT_COLOR) for m in model_names]

    # constrained_layout avoids the tight_layout overflow that caused portrait distortion
    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 2.2), 5),
                             constrained_layout=True)
    title_suffix = f"  (seed={seed})" if seed is not None else ""
    fig.suptitle(f"Test Evaluation Results{title_suffix}", fontsize=13, fontweight="bold")

    short = [m.replace("_PPO", "").replace("_", "\n") for m in model_names]

    # ── subplot 1: average AR ─────────────────────────────────────────────────
    ax1 = axes[0]
    ar_means = [agg_results[m]["ar_mean"] for m in model_names]
    ar_stds  = [agg_results[m]["ar_std"]  for m in model_names]
    # clip P3's inflated AR for display; annotate actual value above bar
    ar_display = [min(v, 1.05) for v in ar_means]
    bars = ax1.bar(x, ar_display, color=colors, alpha=0.8,
                   yerr=ar_stds, capsize=4, ecolor="black", error_kw={"linewidth": 1})
    for bar, v, vd in zip(bars, ar_means, ar_display):
        label = f"{v:.3f}" + (" ▲" if v > 1.05 else "")
        ax1.text(bar.get_x() + bar.get_width() / 2, vd + 0.01,
                 label, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=8.5)
    ax1.set_ylim(0.0, 1.18)
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="AR=1")
    ax1.set_ylabel("Average AR", fontsize=10)
    ax1.set_title("Resource Utilisation (AR)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── subplot 2: success / failure rate ─────────────────────────────────────
    ax2 = axes[1]
    success_rates = [agg_results[m]["success_rate"] for m in model_names]
    failure_rates = [agg_results[m]["failure_rate"] for m in model_names]
    ax2.bar(x, success_rates, color="#2ecc71", alpha=0.85, label="Success")
    ax2.bar(x, failure_rates, bottom=success_rates, color="#e74c3c", alpha=0.85, label="Failure")
    for i, (sr, fr) in enumerate(zip(success_rates, failure_rates)):
        if sr > 0.06:
            ax2.text(i, sr / 2, f"{sr:.0%}", ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white")
        if fr > 0.06:
            ax2.text(i, sr + fr / 2, f"{fr:.0%}", ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=8.5)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Rate", fontsize=10)
    ax2.set_title("Success / Failure Rate", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ── subplot 3: violation rates ─────────────────────────────────────────────
    ax3 = axes[2]
    w = 0.35
    cap_rates  = [agg_results[m]["cap_viol_rate"]  for m in model_names]
    conf_rates = [agg_results[m]["conf_viol_rate"] for m in model_names]
    ax3.bar(x - w / 2, cap_rates,  w, color="#e74c3c", alpha=0.80, label="Cap viol.")
    ax3.bar(x + w / 2, conf_rates, w, color="#f39c12", alpha=0.80, label="Conflict viol.")
    for xi, v in zip(x - w / 2, cap_rates):
        if v > 0.02:
            ax3.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5)
    for xi, v in zip(x + w / 2, conf_rates):
        if v > 0.02:
            ax3.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(short, fontsize=8.5)
    ax3.set_ylim(0.0, 1.10)
    ax3.set_ylabel("Episode Violation Rate", fontsize=10)
    ax3.set_title("Violation Rates", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    path = outdir / "test_results.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")
