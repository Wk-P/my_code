"""
plot_aggregate.py — 从 aggregate_summary.csv 画跨种子均值对比图。

每个 group 单独一张图，4 列子图：AR、Placed、Cap Violations、Conflict Violations。
ILP 用红色虚线标出，RL 方法用柱状图。
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "logs/multi_seed_results/aggregate_summary.csv"
OUT_DIR  = ROOT / "logs/multi_seed_results"

ENV_GROUPS = ["ecu_eq_svc_p", "ecu_gt_svc_p", "ecu_lt_svc_p"]
GROUP_LABELS = {
    "ecu_eq_svc_p": "ECU EQ SVC P (ECUs = Services)",
    "ecu_gt_svc_p": "ECU GT SVC P (ECUs > Services)",
    "ecu_lt_svc_p": "ECU LT SVC P (ECUs < Services)",
}

PROBLEM_ORDER = [
    ("problem3_ppo",            "P3\nPPO\n(ablation)"),
    ("problem4_ppo_mask",       "P4\nPPO-Mask"),
    ("problem5_ppo_lagrangian", "P5\nLagrange"),
    ("problem6_ppo_opt",        "P6\nPPO-Opt"),
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
                "viol":     float(row["viol_rate_avg"]),
            }
    return data


def plot_group(group: str, data: dict, out_path: Path):
    n_problems = len(PROBLEM_ORDER)
    xs         = np.arange(n_problems)
    w          = 0.5
    xlabels    = [lbl for _, lbl in PROBLEM_ORDER]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Multi-Seed Aggregate Results — {GROUP_LABELS[group]}\n(mean across seeds 0–4)",
                 fontsize=13, fontweight="bold")

    # ILP baseline value
    ilp_ar = None
    for prob, _ in PROBLEM_ORDER:
        for (g, p, m), vals in data.items():
            if g == group and p == prob and "ILP" in m:
                ilp_ar = vals["ar"]
                break
        if ilp_ar is not None:
            break

    # Collect RL values per problem
    rl_ar, rl_placed, rl_cap, rl_conflict = [], [], [], []
    for prob, _ in PROBLEM_ORDER:
        matched = [(m, v) for (g, p, m), v in data.items()
                   if g == group and p == prob and "ILP" not in m]
        if matched:
            _, vals = matched[0]
            rl_ar.append(vals["ar"])
            rl_placed.append(vals["placed"])
            rl_cap.append(vals["cap"])
            rl_conflict.append(vals["conflict"])
        else:
            rl_ar.append(0.0)
            rl_placed.append(0.0)
            rl_cap.append(0.0)
            rl_conflict.append(0.0)

    rl_ar       = np.array(rl_ar)
    rl_placed   = np.array(rl_placed)
    rl_cap      = np.array(rl_cap)
    rl_conflict = np.array(rl_conflict)

    def add_labels(ax, bars, fmt=".3f"):
        ylim_top = ax.get_ylim()[1]
        for bar in bars:
            v = bar.get_height()
            text_y = min(v, ylim_top * 0.95) + ylim_top * 0.01
            ax.text(bar.get_x() + bar.get_width() / 2,
                    text_y,
                    f"{v:{fmt}}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=RL_COLOR)

    # ── AR ──────────────────────────────────────────────────────────────
    ax = axes[0]
    # exclude P3 from y-scale if it's an outlier (AR > 2*ILP)
    ar_for_scale = [v for v in rl_ar if v < 2.0] if (ilp_ar and float(np.max(rl_ar)) > 2 * ilp_ar) else list(rl_ar)
    ymax = max(max(ar_for_scale) if ar_for_scale else 0.0, ilp_ar or 0.0) * 1.3 + 0.05
    if ilp_ar is not None:
        ax.axhline(ilp_ar, color=ILP_COLOR, linestyle="--", linewidth=1.5,
                   label=f"ILP = {ilp_ar:.3f}")
    bars = ax.bar(xs, rl_ar, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5, label="RL Method")
    ax.set_ylim(0, ymax)
    add_labels(ax, bars)
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("AR mean")
    ax.set_title("AR (Avg Resource Utilisation)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Placed ──────────────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(xs, rl_placed, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(np.max(rl_placed), 1.0) * 1.25)
    add_labels(ax, bars, fmt=".1f")
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Services Placed (mean)")
    ax.set_title("Placement Completeness", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Cap Violations ──────────────────────────────────────────────────
    ax = axes[2]
    bars = ax.bar(xs, rl_cap, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(np.max(rl_cap), 1.0) * 1.3 + 1)
    add_labels(ax, bars, fmt=".1f")
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Cap Violations (mean)")
    ax.set_title("Capacity Violations", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Conflict Violations ─────────────────────────────────────────────
    ax = axes[3]
    bars = ax.bar(xs, rl_conflict, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(np.max(rl_conflict), 1.0) * 1.3 + 1)
    add_labels(ax, bars, fmt=".1f")
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Conflict Violations (mean)")
    ax.set_title("Conflict Violations", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    data = load_csv(CSV_PATH)
    for group in ENV_GROUPS:
        out_path = OUT_DIR / f"aggregate_plot_{group}.png"
        plot_group(group, data, out_path)


if __name__ == "__main__":
    main()
