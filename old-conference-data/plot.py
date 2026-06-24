import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# Quick-validation constraint-mode variants (single seed, 600k steps) — LT only.
# Source: old-research-code/ecu_lt_svc_p/compare_dqn_constraints_results.json
DQN_CONSTRAINT_MODES_JSON = HERE.parent / "old-research-code" / "ecu_lt_svc_p" / "compare_dqn_constraints_results.json"
ALGO_TO_METHOD = {"dqn": "DQN", "ddqn": "DDQN"}
REPLACEMENT_MODE = "hard"  # swap in the hard-terminate retrain for DQN/DDQN in LT

SETTINGS = {
    "lt": "LT (ECU < SVC)",
    "eq": "EQ (ECU = SVC)",
    "gt": "GT (ECU > SVC)",
}

METHOD_ORDER = [
    "ILP",
    "MaskablePPO",
    "Lagrange PPO",
    "PPO+Repair",
    "DQN",
    "DDQN",
]

# METHOD_ORDER holds the display names (no "(P4)"/"(P5)"/"(P6)" codes); this maps
# each one to the full string actually stored in results.csv's "method" column.
METHOD_CSV_NAME = {
    "ILP":          "ILP (Optimal)",
    "MaskablePPO":  "MaskablePPO (P4)",
    "Lagrange PPO": "Lagrange PPO (P5)",
    "PPO+Repair":   "PPO+Repair (P6)",
    "DQN":          "DQN",
    "DDQN":         "DDQN",
}

N_TEST = 40  # test episodes per seed, same for all settings

# 3 grouped bars per method
BAR_LABELS  = ["Average Utilization Rate (AR)", "Capacity Violation Rate", "Conflict Violation Rate"]
BAR_COLORS  = ["#4292c6", "#f16913", "#74c476"]
BAR_WIDTH   = 0.25
N_BARS      = 3


def load_dqn_constraint_replacements() -> dict:
    """DQN/DDQN retrained with hard-termination constraint handling (LT only,
    single seed, 600k steps — quick validation, not the full 5-seed protocol).
    Returns {method_name: {ar_mean_avg, cap_viol_rate, conflict_viol_rate}}."""
    with open(DQN_CONSTRAINT_MODES_JSON) as f:
        rows = json.load(f)
    return {
        ALGO_TO_METHOD[r["algo"]]: {
            "ar_mean_avg":        r["ar_mean"],
            "cap_viol_rate":      r["cap_viol_rate"],
            "conflict_viol_rate": r["conflict_viol_rate"],
        }
        for r in rows if r["mode"] == REPLACEMENT_MODE
    }


def load_setting(setting: str) -> pd.DataFrame:
    df = pd.read_csv(HERE / setting / "results.csv")
    csv_names = list(METHOD_CSV_NAME.values())
    df = df[df["method"].isin(csv_names)].copy()
    csv_to_display = {v: k for k, v in METHOD_CSV_NAME.items()}
    df["method"] = df["method"].map(csv_to_display)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)
    df["cap_viol_rate"]     = df["cap_viol_avg"]     / N_TEST
    df["conflict_viol_rate"] = df["conflict_viol_avg"] / N_TEST
    df["method"] = df["method"].astype(str)
    df = df[["method", "ar_mean_avg", "cap_viol_rate", "conflict_viol_rate"]]

    if setting == "lt":
        replacements = load_dqn_constraint_replacements()
        for method, vals in replacements.items():
            df.loc[df["method"] == method, ["ar_mean_avg", "cap_viol_rate", "conflict_viol_rate"]] = (
                vals["ar_mean_avg"], vals["cap_viol_rate"], vals["conflict_viol_rate"]
            )
    return df


fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=False)

for ax, (setting, title) in zip(axes, SETTINGS.items()):
    df = load_setting(setting)
    methods = df["method"].tolist()
    n = len(methods)
    x = np.arange(n)

    offsets = [-BAR_WIDTH, 0, BAR_WIDTH]
    cols    = [df["ar_mean_avg"], df["cap_viol_rate"], df["conflict_viol_rate"]]

    for offset, vals, color, label in zip(offsets, cols, BAR_COLORS, BAR_LABELS):
        ax.bar(x + offset, vals, BAR_WIDTH * 0.9,
               color=color, edgecolor="white", linewidth=0.4, label=label)
        for xi, v in zip(x + offset, vals):
            if v > 0.005:
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel("Rate (0-1)", fontsize=13)
    ax.set_xlabel("Method", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=11.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)  # headroom so the top legend doesn't overlap bars

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=13, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))

fig.suptitle("Average Utilization Rate vs Constraint Violation Rates (Capacity & Conflict) by Method",
             fontsize=17, fontweight="bold", y=1.16)
plt.tight_layout()
out = HERE / "bar_comparison.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
