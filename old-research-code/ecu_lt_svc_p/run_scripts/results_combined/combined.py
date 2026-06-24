import os
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROBLEM_ORDER = [
    ("problem3_ppo",            "P3\nPPO"),
    ("problem4_ppo_mask",       "P4\nPPO-Mask"),
    ("problem5_ppo_lagrangian", "P5\nPPO-Lagrange"),
    ("problem6_ppo_opt",        "P6\nPPO-Opt"),
    ("problem_dqn",             "DQN"),
    ("problem_ddqn",            "DDQN"),
]

RL_COLOR = "#1f77b4"


def load_all():
    records = {}
    for folder, _ in PROBLEM_ORDER:
        csv_path = os.path.join(BASE_DIR, folder, "summary.csv")
        if not os.path.exists(csv_path):
            print(f"[warn] missing: {csv_path}")
            continue
        records[folder] = pd.read_csv(csv_path)
    return records


def _get_rows(df):
    ilp_df = df[df["method"].str.contains("ILP|Optimal", regex=True, na=False)]
    rl_df  = df[~df["method"].str.contains("ILP|Optimal", regex=True, na=False)]
    return (
        ilp_df.iloc[0] if not ilp_df.empty else None,
        rl_df.iloc[0]  if not rl_df.empty  else None,
    )


def _val(row, col, default=0.0):
    if row is None:
        return default
    v = row.get(col, default)
    return float(v) if pd.notna(v) else default


def plot_combined(records):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Combined Results: P3–P6 PPO Variants + DQN vs ILP Optimal",
                 fontsize=13, fontweight="bold")

    n  = len(PROBLEM_ORDER)
    xs = np.arange(n)
    w  = 0.4

    keys = ["ilp_ar", "rl_ar", "rl_placed", "rl_cap_viol", "rl_conflict_viol"]
    data = {k: [] for k in keys}

    for folder, _ in PROBLEM_ORDER:
        df = records.get(folder, pd.DataFrame())
        ilp, rl = _get_rows(df) if not df.empty else (None, None)
        data["ilp_ar"].append(_val(ilp, "ar_mean"))
        data["rl_ar"].append(_val(rl,  "ar_mean"))
        data["rl_placed"].append(_val(rl, "placed_mean"))
        data["rl_cap_viol"].append(_val(rl, "cap_viol_total"))
        data["rl_conflict_viol"].append(_val(rl, "conflict_viol_total"))

    for k in data:
        data[k] = np.array(data[k])

    xlabels  = [lbl for _, lbl in PROBLEM_ORDER]
    valid_ilp = data["ilp_ar"][data["ilp_ar"] > 0]
    ilp_mean  = float(np.mean(valid_ilp)) if len(valid_ilp) else 0.0

    # ── Subplot 1: AR Comparison ──────────────────────────────────────────
    ax = axes[0]
    ax.axhline(ilp_mean, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"ILP mean={ilp_mean:.3f}")
    b_rl = ax.bar(xs, data["rl_ar"], width=w,
                  color=RL_COLOR, edgecolor="black", linewidth=0.5, label="RL Method")

    for bar, v in zip(b_rl, data["rl_ar"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color=RL_COLOR)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Average Resource Utilisation (AR)")
    ax.set_title("AR Comparison: ILP vs Method", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Subplot 2: Capacity Violations ───────────────────────────────────
    ax = axes[1]
    b_cap = ax.bar(xs, data["rl_cap_viol"], width=w,
                   color=RL_COLOR, edgecolor="black", linewidth=0.5)

    cap_max = max(np.max(data["rl_cap_viol"]), 1.0)
    for bar, v in zip(b_cap, data["rl_cap_viol"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + cap_max * 0.02,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Total Capacity Violations")
    ax.set_title("Capacity Violations (cap_viol_total)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, cap_max * 1.3 + 1)

    # ── Subplot 3: Conflict Violations ────────────────────────────────────
    ax = axes[2]
    b_conf = ax.bar(xs, data["rl_conflict_viol"], width=w,
                    color=RL_COLOR, edgecolor="black", linewidth=0.5)

    conf_max = max(np.max(data["rl_conflict_viol"]), 1.0)
    for bar, v in zip(b_conf, data["rl_conflict_viol"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + conf_max * 0.02,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Total Conflict Violations")
    ax.set_title("Conflict Violations (conflict_viol_total)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, conf_max * 1.3 + 1)

    # ── Subplot 4: Placement Completeness ─────────────────────────────────
    ax = axes[3]
    b_rl3 = ax.bar(xs, data["rl_placed"], width=w,
                   color=RL_COLOR, edgecolor="black", linewidth=0.5)

    ymax = max(np.max(data["rl_placed"]), 1.0)
    offset = ymax * 0.015 + 0.05
    for bar, v in zip(b_rl3, data["rl_placed"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Services Placed")
    ax.set_title("Placement Completeness", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, ymax * 1.2)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "combined_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def print_summary_table(df):
    print("\n===== Combined Summary =====")
    cols = ["problem", "method", "ar_mean", "ar_std", "placed_mean",
            "viol_rate", "cap_viol_total", "conflict_viol_total"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False, float_format="%.4f"))

    print("\n===== Best RL Agent per Problem =====")
    rl = df[~df["method"].str.contains("ILP|Optimal", regex=True, na=False)]
    if not rl.empty:
        best = rl.loc[rl.groupby("folder")["ar_mean"].idxmax()]
        print(best[available].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    records = load_all()

    flat_dfs = []
    for folder, label in PROBLEM_ORDER:
        if folder in records:
            df = records[folder].copy()
            df["problem"] = label.replace("\n", " ")
            df["folder"]  = folder
            flat_dfs.append(df)
    if flat_dfs:
        print_summary_table(pd.concat(flat_dfs, ignore_index=True))

    plot_combined(records)
    print("\nDone.")
