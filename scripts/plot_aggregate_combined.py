"""
plot_aggregate_combined.py — 3×4 combined figure across all ECU groups.

Row order : LT → EQ → GT
Columns   : Average Resource Utilization | Sum of Placed Services |
            Average Capacity Violations  | Average Privacy Violations
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT     = Path(__file__).parent.parent
CSV_PATH = ROOT / "logs/multi_seed_results/aggregate_summary.csv"
OUT_DIR  = ROOT / "logs/multi_seed_results"

# ── row / column definitions ──────────────────────────────────────────────────
ROW_GROUPS = ["ecu_lt_svc_p", "ecu_eq_svc_p", "ecu_gt_svc_p"]
ROW_LABELS = ["LT (ECUs < Services)", "EQ (ECUs = Services)", "GT (ECUs > Services)"]

COL_TITLES = [
    "Average Resource Utilization",
    "Sum of Placed Services",
    "Average Capacity Violations",
    "Average Privacy Violations",
]

PROBLEM_ORDER = [
    ("problem3_ppo",            "PPO\n(ablation)"),
    ("problem4_ppo_mask",       "PPO-Mask"),
    ("problem5_ppo_lagrangian", "Lagrange"),
    ("problem6_ppo_opt",        "PPO-Opt"),
    ("problem_dqn",             "DQN"),
    ("problem_ddqn",            "DDQN"),
]

RL_COLOR  = "#1f77b4"
ILP_COLOR = "#d62728"


def load_csv(path: Path):
    data = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["group"], row["problem"], row["method"])
            data[key] = {
                "ar":       float(row["ar_mean_avg"]),
                "placed":   float(row["placed_mean_avg"]),
                "cap":      float(row["cap_viol_avg"]),
                "conflict": float(row["conflict_viol_avg"]),
            }
    return data


def get_row_data(group: str, data: dict):
    rl_ar, rl_placed, rl_cap, rl_conflict = [], [], [], []
    ilp_ar = None

    for prob, _ in PROBLEM_ORDER:
        rl_vals, ilp_vals = None, None
        for (g, p, m), v in data.items():
            if g != group or p != prob:
                continue
            if "ILP" in m or "Optimal" in m:
                ilp_vals = v
            else:
                rl_vals = v
        if ilp_ar is None and ilp_vals is not None:
            ilp_ar = ilp_vals["ar"]
        rl_ar.append(rl_vals["ar"]       if rl_vals else 0.0)
        rl_placed.append(rl_vals["placed"]   if rl_vals else 0.0)
        rl_cap.append(rl_vals["cap"]      if rl_vals else 0.0)
        rl_conflict.append(rl_vals["conflict"] if rl_vals else 0.0)

    return (np.array(rl_ar), np.array(rl_placed),
            np.array(rl_cap), np.array(rl_conflict), ilp_ar)


def add_bar_labels(ax, bars, fmt=".3f"):
    ylim_top = ax.get_ylim()[1]
    for bar in bars:
        v = bar.get_height()
        text_y = min(v, ylim_top * 0.95) + ylim_top * 0.01
        ax.text(bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{v:{fmt}}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", color=RL_COLOR)


def draw_row(axes_row, rl_ar, rl_placed, rl_cap, rl_conflict, ilp_ar,
             xs, w, xlabels, show_col_titles=False):
    """Fill one row of 4 axes with bar charts."""

    # ── col 0: Average Resource Utilization (AR capped at 1.0) ───────────
    ax       = axes_row[0]
    ymax     = 1.05
    clip_top = 1.0
    if show_col_titles:
        ax.set_title(COL_TITLES[0], fontsize=11, fontweight="bold", pad=8)
    if ilp_ar is not None:
        ax.axhline(ilp_ar, color=ILP_COLOR, linestyle="--", linewidth=1.5,
                   label=f"ILP={ilp_ar:.3f}")
        ax.legend(fontsize=8, loc="upper right")
    rl_ar_disp = np.minimum(rl_ar, clip_top)
    bars = ax.bar(xs, rl_ar_disp, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, ymax)
    for bar, true_val in zip(bars, rl_ar):
        clipped = true_val > clip_top
        label   = f"↑{true_val:.3f}" if clipped else f"{true_val:.3f}"
        color   = "#d62728" if clipped else RL_COLOR
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.01,
                label, ha="center", va="bottom",
                fontsize=7, fontweight="bold", color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Safe RL Models", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── col 1: Sum of Placed Services ────────────────────────────────────
    ax = axes_row[1]
    if show_col_titles:
        ax.set_title(COL_TITLES[1], fontsize=11, fontweight="bold", pad=8)
    bars = ax.bar(xs, rl_placed, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(float(np.max(rl_placed)), 1.0) * 1.25)
    add_bar_labels(ax, bars, fmt=".1f")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Safe RL Models", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── col 2: Average Capacity Violations ───────────────────────────────
    ax = axes_row[2]
    if show_col_titles:
        ax.set_title(COL_TITLES[2], fontsize=11, fontweight="bold", pad=8)
    bars = ax.bar(xs, rl_cap, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(float(np.max(rl_cap)), 1.0) * 1.35 + 1)
    add_bar_labels(ax, bars, fmt=".1f")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Safe RL Models", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── col 3: Average Privacy Violations ────────────────────────────────
    ax = axes_row[3]
    if show_col_titles:
        ax.set_title(COL_TITLES[3], fontsize=11, fontweight="bold", pad=8)
    bars = ax.bar(xs, rl_conflict, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(float(np.max(rl_conflict)), 1.0) * 1.35 + 1)
    add_bar_labels(ax, bars, fmt=".1f")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Safe RL Models", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def save_fig(fig, stem: str):
    """Save as both PDF (vector) and PNG (600 DPI)."""
    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_path}")


def plot_single(group: str, row_label: str, data: dict):
    rl_ar, rl_placed, rl_cap, rl_conflict, ilp_ar = get_row_data(group, data)
    xs      = np.arange(len(PROBLEM_ORDER))
    xlabels = [lbl for _, lbl in PROBLEM_ORDER]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(
        f"Performance Evaluation of Safe RL Models Against ILP Optimal Baseline — {row_label}",
        fontsize=13, fontweight="bold")

    draw_row(axes, rl_ar, rl_placed, rl_cap, rl_conflict, ilp_ar,
             xs, w=0.5, xlabels=xlabels, show_col_titles=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, f"aggregate_plot_{group}")
    plt.close()


def plot_combined(data: dict):
    xs      = np.arange(len(PROBLEM_ORDER))
    xlabels = [lbl for _, lbl in PROBLEM_ORDER]

    fig, axes = plt.subplots(3, 4, figsize=(28, 18))
    fig.suptitle(
        "Performance Evaluation of Safe RL Models Against ILP Optimal Baseline",
        fontsize=14, fontweight="bold")

    for row_idx, (group, row_label) in enumerate(zip(ROW_GROUPS, ROW_LABELS)):
        rl_ar, rl_placed, rl_cap, rl_conflict, ilp_ar = get_row_data(group, data)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=11, fontweight="bold",
                                    rotation=90, labelpad=8)
        draw_row(axes[row_idx], rl_ar, rl_placed, rl_cap, rl_conflict, ilp_ar,
                 xs, w=0.5, xlabels=xlabels, show_col_titles=(row_idx == 0))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "aggregate_plot_combined")
    plt.close()


def main():
    data = load_csv(CSV_PATH)
    for group, row_label in zip(ROW_GROUPS, ROW_LABELS):
        plot_single(group, row_label, data)
    plot_combined(data)


if __name__ == "__main__":
    main()
