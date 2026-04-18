import os
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

RL_COLORS = {
    "problem3_ppo":            "#1f77b4",
    "problem4_ppo_mask":       "#2ca02c",
    "problem5_ppo_lagrangian": "#ff7f0e",
    "problem6_ppo_opt":        "#9467bd",
    "problem_dqn":             "#17becf",
    "problem_ddqn":            "#bcbd22",
}

RL_LABELS = {
    "problem3_ppo":            "P3 PPO",
    "problem4_ppo_mask":       "P4 PPO-Mask",
    "problem5_ppo_lagrangian": "P5 PPO-Lagrange",
    "problem6_ppo_opt":        "P6 PPO-Opt",
    "problem_dqn":             "DQN",
    "problem_ddqn":            "DDQN",
}


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
    ilp_df  = df[df["method"].str.contains("ILP|Optimal", regex=True, na=False)]
    rand_df = df[df["method"].str.contains("Random",      regex=True, na=False)]
    rl_df   = df[~df["method"].str.contains("ILP|Optimal|Random", regex=True, na=False)]
    return (
        ilp_df.iloc[0]  if not ilp_df.empty  else None,
        rand_df.iloc[0] if not rand_df.empty else None,
        rl_df.iloc[0]   if not rl_df.empty   else None,
    )


def _val(row, col, default=0.0):
    if row is None:
        return default
    v = row.get(col, default)
    return float(v) if pd.notna(v) else default


def plot_combined(records):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Combined Results: P3–P6 PPO Variants + DQN vs ILP Optimal",
                 fontsize=13, fontweight="bold")

    n  = len(PROBLEM_ORDER)
    xs = np.arange(n)
    w  = 0.25

    keys = ["ilp_ar", "ilp_std", "ilp_placed",
            "rand_ar", "rand_std", "rand_placed", "rand_viol",
            "rl_ar",  "rl_std",  "rl_placed",  "rl_viol"]
    data = {k: [] for k in keys}

    for folder, _ in PROBLEM_ORDER:
        df = records.get(folder, pd.DataFrame())
        ilp, rand, rl = _get_rows(df) if not df.empty else (None, None, None)
        data["ilp_ar"].append(_val(ilp,  "ar_mean"))
        data["ilp_std"].append(_val(ilp, "ar_std"))
        data["ilp_placed"].append(_val(ilp, "placed_mean"))
        data["rand_ar"].append(_val(rand, "ar_mean"))
        data["rand_std"].append(_val(rand, "ar_std"))
        data["rand_placed"].append(_val(rand, "placed_mean"))
        data["rand_viol"].append(_val(rand, "viol_rate"))
        data["rl_ar"].append(_val(rl,  "ar_mean"))
        data["rl_std"].append(_val(rl, "ar_std"))
        data["rl_placed"].append(_val(rl, "placed_mean"))
        data["rl_viol"].append(_val(rl,  "viol_rate"))

    for k in data:
        data[k] = np.array(data[k])

    rl_colors = [RL_COLORS[f] for f, _ in PROBLEM_ORDER]
    xlabels   = [lbl          for _, lbl in PROBLEM_ORDER]
    valid_ilp  = data["ilp_ar"][data["ilp_ar"] > 0]
    ilp_mean   = float(np.mean(valid_ilp)) if len(valid_ilp) else 0.0

    # ── Subplot 1: AR Comparison ──────────────────────────────────────────
    ax = axes[0]
    ax.axhline(ilp_mean, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"ILP mean={ilp_mean:.3f}")
    b_ilp  = ax.bar(xs - w, data["ilp_ar"],  width=w, yerr=data["ilp_std"],
                    color="#d62728", capsize=3, edgecolor="black", linewidth=0.5,
                    label="ILP (Optimal)")
    b_rand = ax.bar(xs,     data["rand_ar"], width=w, yerr=data["rand_std"],
                    color="#aec7e8", capsize=3, edgecolor="black", linewidth=0.5,
                    label="Random Baseline")
    b_rl   = ax.bar(xs + w, data["rl_ar"],   width=w, yerr=data["rl_std"],
                    color=rl_colors, capsize=3, edgecolor="black", linewidth=0.5)
    for folder, lbl in PROBLEM_ORDER:
        ax.bar([], [], color=RL_COLORS[folder], label=RL_LABELS[folder])

    for bar, v, e, c in zip(b_rl, data["rl_ar"], data["rl_std"], rl_colors):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color=c)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Average Resource Utilisation (AR)")
    ax.set_title("AR Comparison: ILP vs Random vs Method", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Subplot 2: Constraint Violation Rate ──────────────────────────────
    ax = axes[1]
    ax.bar(xs - w / 2, data["rand_viol"], width=w,
           color="#aec7e8", edgecolor="black", linewidth=0.5, label="Random")
    b_rl2 = ax.bar(xs + w / 2, data["rl_viol"], width=w,
                   color=rl_colors, edgecolor="black", linewidth=0.5)
    for folder, lbl in PROBLEM_ORDER:
        ax.bar([], [], color=RL_COLORS[folder], label=RL_LABELS[folder])

    for bar, v in zip(b_rl2, data["rl_viol"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Violation Rate")
    ax.set_title("Constraint Violation Rate", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    vmax = max(np.max(data["rand_viol"]), np.max(data["rl_viol"]))
    ax.set_ylim(0, vmax * 1.3 + 0.05)

    # ── Subplot 3: Placement Completeness ─────────────────────────────────
    ax = axes[2]
    b_ilp3  = ax.bar(xs - w, data["ilp_placed"],  width=w,
                     color="#d62728", edgecolor="black", linewidth=0.5, label="ILP (Optimal)")
    b_rand3 = ax.bar(xs,     data["rand_placed"], width=w,
                     color="#aec7e8", edgecolor="black", linewidth=0.5, label="Random Baseline")
    b_rl3   = ax.bar(xs + w, data["rl_placed"],   width=w,
                     color=rl_colors, edgecolor="black", linewidth=0.5)
    for folder, lbl in PROBLEM_ORDER:
        ax.bar([], [], color=RL_COLORS[folder], label=RL_LABELS[folder])

    ymax = max(np.max(data["ilp_placed"]), np.max(data["rand_placed"]),
               np.max(data["rl_placed"]), 1.0)
    offset = ymax * 0.015 + 0.05
    for bar, v in zip(b_rl3, data["rl_placed"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, v in zip(b_ilp3, data["ilp_placed"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                color="#d62728")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Services Placed")
    ax.set_title("Placement Completeness", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2)
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
    rl = df[~df["method"].str.contains("ILP|Optimal|Random", regex=True, na=False)]
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
