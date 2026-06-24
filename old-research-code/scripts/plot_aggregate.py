"""
plot_aggregate.py — 从 aggregate_summary.csv 画跨种子均值对比图。

每个 group 一行，4 列子图：AR、Placed、Cap Violations、Conflict Violations。
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
OUT_PATH = ROOT / "logs/multi_seed_results/aggregate_plot.png"

ENV_GROUPS = ["ecu_eq_svc_p", "ecu_gt_svc_p", "ecu_lt_svc_p"]
GROUP_LABELS = {"ecu_eq_svc_p": "ECU EQ SVC P", "ecu_gt_svc_p": "ECU GT SVC P", "ecu_lt_svc_p": "ECU LT SVC P"}

PROBLEM_ORDER = [
    ("problem3_ppo",            "P3\nPPO"),
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


def main():
    data = load_csv(CSV_PATH)

    n_groups   = len(ENV_GROUPS)
    n_problems = len(PROBLEM_ORDER)
    xs         = np.arange(n_problems)
    w          = 0.5
    xlabels    = [lbl for _, lbl in PROBLEM_ORDER]

    fig, axes = plt.subplots(n_groups, 4, figsize=(22, 5 * n_groups))
    fig.suptitle("Multi-Seed Aggregate Results (mean across seeds 0–4)",
                 fontsize=14, fontweight="bold", y=1.01)

    for row_idx, group in enumerate(ENV_GROUPS):
        ar_ilp = conflict = cap = placed = rl_ar = None

        # 取 ILP 值（同 group 内各 problem ILP 值相同，取第一个）
        ilp_ar = None
        for prob, _ in PROBLEM_ORDER:
            for (g, p, m), vals in data.items():
                if g == group and p == prob and "ILP" in m:
                    ilp_ar = vals["ar"]
                    break
            if ilp_ar is not None:
                break

        # 收集各 problem 的 RL 数据
        rl_ar       = []
        rl_placed   = []
        rl_cap      = []
        rl_conflict = []

        for prob, _ in PROBLEM_ORDER:
            matched = [(m, v) for (g, p, m), v in data.items()
                       if g == group and p == prob and "ILP" not in m]
            if matched:
                m_name, vals = matched[0]
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
            for bar in bars:
                v = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + ax.get_ylim()[1] * 0.01,
                        f"{v:{fmt}}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color=RL_COLOR)

        # ── AR ──────────────────────────────────────────────────────────
        ax = axes[row_idx, 0]
        if ilp_ar is not None:
            ax.axhline(ilp_ar, color=ILP_COLOR, linestyle="--", linewidth=1.2,
                       label=f"ILP={ilp_ar:.3f}")
        bars = ax.bar(xs, rl_ar, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, max(np.max(rl_ar), ilp_ar or 0) * 1.25 + 0.05)
        add_labels(ax, bars)
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("AR mean"); ax.set_title(f"{GROUP_LABELS[group]}\nAR", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # ── Placed ──────────────────────────────────────────────────────
        ax = axes[row_idx, 1]
        bars = ax.bar(xs, rl_placed, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, max(np.max(rl_placed), 1.0) * 1.25)
        add_labels(ax, bars, fmt=".1f")
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Placed mean"); ax.set_title("Placed", fontsize=9, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # ── Cap Violations ──────────────────────────────────────────────
        ax = axes[row_idx, 2]
        bars = ax.bar(xs, rl_cap, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, max(np.max(rl_cap), 1.0) * 1.3 + 1)
        add_labels(ax, bars, fmt=".1f")
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Cap Viol"); ax.set_title("Capacity Violations", fontsize=9, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # ── Conflict Violations ─────────────────────────────────────────
        ax = axes[row_idx, 3]
        bars = ax.bar(xs, rl_conflict, width=w, color=RL_COLOR, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, max(np.max(rl_conflict), 1.0) * 1.3 + 1)
        add_labels(ax, bars, fmt=".1f")
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Conflict Viol"); ax.set_title("Conflict Violations", fontsize=9, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
