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
}

_DEFAULT_COLOR = "#aaaaaa"


def _moving_avg(vals: list[float], w: int) -> tuple[np.ndarray, int]:
    if len(vals) < w:
        return np.array(vals, dtype=float), 0
    kernel = np.ones(w) / w
    smoothed = np.convolve(vals, kernel, mode="valid")
    return smoothed, (w - 1) // 2


def plot_training_curves(
    training_data: dict[str, dict],
    outdir: Path,
    smooth_w: int = 2000,
    seed: int | None = None,
) -> None:
    """
    training_data: {model_name: {"episode_nums": [...], "episode_rewards": [...]}}
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    title = "Training Reward vs Episode"
    if seed is not None:
        title += f"  (seed={seed})"

    for model_name, data in training_data.items():
        ep_nums   = np.array(data["episode_nums"],   dtype=float)
        ep_rews   = np.array(data["episode_rewards"], dtype=float)
        color     = MODEL_COLORS.get(model_name, _DEFAULT_COLOR)

        ax.plot(ep_nums, ep_rews, color=color, alpha=0.15, linewidth=0.6)

        sm, off = _moving_avg(ep_rews, smooth_w)
        if len(sm) > 0:
            ax.plot(
                ep_nums[off: off + len(sm)],
                sm,
                color=color,
                linewidth=2.0,
                label=model_name,
            )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
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
    Produces a 3-subplot figure.
    """
    model_names = list(agg_results.keys())
    n = len(model_names)
    x = np.arange(n)
    colors = [MODEL_COLORS.get(m, _DEFAULT_COLOR) for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    title_suffix = f"  (seed={seed})" if seed is not None else ""
    fig.suptitle(f"Test Evaluation Results{title_suffix}", fontsize=13, fontweight="bold")

    # ── subplot 1: average AR ────────────────────────────────────────────────
    ax1 = axes[0]
    ar_means = [agg_results[m]["ar_mean"] for m in model_names]
    ar_stds  = [agg_results[m]["ar_std"]  for m in model_names]
    bars = ax1.bar(x, ar_means, color=colors, alpha=0.75,
                   yerr=ar_stds, capsize=4, ecolor="black")
    for bar, v in zip(bars, ar_means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, v + 0.01,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Average AR", fontsize=11)
    ax1.set_title("Average Resource Utilisation", fontsize=11)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # ── subplot 2: success / failure rate ────────────────────────────────────
    ax2 = axes[1]
    success_rates = [agg_results[m]["success_rate"] for m in model_names]
    failure_rates = [agg_results[m]["failure_rate"] for m in model_names]
    ax2.bar(x, success_rates, color="#2ecc71", alpha=0.8, label="Success")
    ax2.bar(x, failure_rates, bottom=success_rates, color="#e74c3c", alpha=0.8, label="Failure")
    for i, (sr, fr) in enumerate(zip(success_rates, failure_rates)):
        ax2.text(i, sr / 2, f"{sr:.1%}", ha="center", va="center",
                 fontsize=8, fontweight="bold", color="white")
        if fr > 0.04:
            ax2.text(i, sr + fr / 2, f"{fr:.1%}", ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Rate", fontsize=11)
    ax2.set_title("Success / Failure Rate\n(100% placed + no violation)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ── subplot 3: violation rates ────────────────────────────────────────────
    ax3 = axes[2]
    w = 0.35
    cap_rates  = [agg_results[m]["cap_viol_rate"]  for m in model_names]
    conf_rates = [agg_results[m]["conf_viol_rate"] for m in model_names]
    bars_cap  = ax3.bar(x - w / 2, cap_rates,  w, color="#e74c3c", alpha=0.75, label="Cap violation")
    bars_conf = ax3.bar(x + w / 2, conf_rates, w, color="#f39c12", alpha=0.75, label="Conflict violation")
    for bar, v in zip(bars_cap, cap_rates):
        if v > 0.02:
            ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f"{v:.1%}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars_conf, conf_rates):
        if v > 0.02:
            ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f"{v:.1%}", ha="center", va="bottom", fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax3.set_ylim(0.0, 1.05)
    ax3.set_ylabel("Episode Violation Rate", fontsize=11)
    ax3.set_title("Violation Rates\n(episodes with ≥1 violation / total episodes)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "test_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")
