from os import read
from pathlib import Path
import sys, time, csv
import numpy as np
import json 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).parent
ROOT_PATH = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_PATH))


# read csv files from paths

PATH_LIST = [
    HERE / "problem3_ppo" / "summary.csv",
    HERE / "problem4_ppo_mask" / "summary.csv",
    HERE / "problem5_ppo_lagrangian" / "summary.csv",
    HERE / "problem6_ppo_opt" / "summary.csv",
    HERE / "dqn" / "summary.csv",
]

ILP_DATA_PATH = HERE / "ilp_cache.json"

head_text_width = 10
print(f"{'HERE:':<{head_text_width}} {HERE}")
print(f"{'ROOT_PATH:':<{head_text_width}} {ROOT_PATH}")



"""This script combines results from all problems (3-6 PPO variants + DQN) and ILP into a single CSV file for easier comparison and plotting."""

def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def read_csv(path: Path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)
    

def plot_combined_results(data_list, ilp_data, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    (ax1, ax2), (ax3, ax4) = axes

    problem_names  = ["P3\nPPO", "P4\nPPO-Mask", "P5\nPPO-Lagrange", "P6\nPPO-Opt", "DQN"]
    pal_ilp    = "#e74c3c"
    pal_random = "#95a5a6"
    pal_method = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e67e22"]
    pal_method_2 = ["#3498db" for _ in range(len(pal_method))]   # same colors as pal_method, but DQN is orange
    x  = np.arange(len(problem_names))
    bw = 0.26  # bar width

    # ── Parse each problem's three rows ────────────────────────────────────────
    ilp_ar, random_ar, method_ar   = [], [], []
    random_std, method_std         = [], []
    random_viol, method_viol       = [], []
    placed_mean_list               = []
    dqn_flags                      = []   # True → DQN (viol_rate), False → PPO (viol_per_ep)

    for data in data_list:
        def _row(kw):
            return next((r for r in data if kw.lower() in r["method"].lower()), {})
        ilp  = _row("ILP")
        rand = _row("Random")
        mth  = next((r for r in data
                     if "ILP" not in r["method"] and "Random" not in r["method"]), {})

        ilp_ar.append(float(ilp.get("ar_mean", 0)))
        random_ar.append(float(rand.get("ar_mean", 0)))
        method_ar.append(float(mth.get("ar_mean", 0)))
        random_std.append(float(rand.get("ar_std", 0)))
        method_std.append(float(mth.get("ar_std", 0)))

        if "viol_per_ep" in rand:
            random_viol.append(float(rand["viol_per_ep"]))
            method_viol.append(float(mth.get("viol_per_ep", 0)))
            dqn_flags.append(False)
        else:
            random_viol.append(float(rand.get("viol_rate", 0)))
            method_viol.append(float(mth.get("viol_rate", 0)))
            dqn_flags.append(True)

        placed_mean_list.append(float(mth.get("placed_mean", 10)))

    # ══════════════════════════════════════════════════════════════════════════
    # ax1 — AR mean: ILP / Random / Method grouped bar
    # ══════════════════════════════════════════════════════════════════════════
    ax1.bar(x - bw, ilp_ar,    bw, color=pal_ilp,   alpha=0.85, label="ILP (Optimal)", zorder=3)
    ax1.bar(x,      random_ar, bw, color=pal_random, alpha=0.85, label="Random Baseline",
            yerr=random_std, capsize=4, ecolor="#7f8c8d", zorder=3)
    for xi, (ar, std, color) in enumerate(zip(method_ar, method_std, pal_method_2)):
        ax1.bar(xi + bw, ar, bw, color=color, alpha=0.85,
                yerr=std, capsize=4, ecolor="#555", zorder=3,
                label="PPO / DQN" if xi == 0 else "_")
        ax1.text(xi + bw, ar + std + 0.015, f"{ar:.3f}",
                 ha="center", fontsize=8, fontweight="bold", color=color)

    mean_ilp = float(np.mean(ilp_ar))
    ax1.axhline(mean_ilp, color=pal_ilp, linestyle="--", linewidth=1.2,
                alpha=0.55, label=f"ILP mean={mean_ilp:.3f}")
    ax1.set_xticks(x); ax1.set_xticklabels(problem_names)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Average Resource Utilisation (AR)")
    ax1.set_title("AR Comparison: ILP vs Random vs Method", fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # ══════════════════════════════════════════════════════════════════════════
    # ax2 — Constraint violations
    #   P3-P6: viol_per_ep (count, left axis)
    #   DQN:   viol_rate   (0-1, right twin axis)
    # ══════════════════════════════════════════════════════════════════════════
    ppo_x  = [i for i, d in enumerate(dqn_flags) if not d]
    dqn_x  = [i for i, d in enumerate(dqn_flags) if d]

    for xi in ppo_x:
        ax2.bar(xi - bw/2, random_viol[xi], bw, color=pal_random, alpha=0.8,
                label="Random" if xi == ppo_x[0] else "_", zorder=3)
        ax2.bar(xi + bw/2, method_viol[xi], bw, color=pal_method_2, alpha=0.8,
                label="PPO" if xi == ppo_x[0] else "_", zorder=3)

    ax2.set_xticks(x); ax2.set_xticklabels(problem_names)
    ax2.set_ylabel("Violations / Episode  (P3–P6)", color="#2c3e50")
    ax2.set_title("Constraint Violations per Episode", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    # DQN twin axis (viol_rate)
    ax2b = ax2.twinx()
    for xi in dqn_x:
        ax2b.bar(xi - bw/2, random_viol[xi], bw, color=pal_random, alpha=0.8,
                 label="Random (rate)" if xi == dqn_x[0] else "_", zorder=3, hatch="//")
        ax2b.bar(xi + bw/2, method_viol[xi], bw, color=pal_method_2[xi], alpha=0.8,
                 label="DQN (rate)" if xi == dqn_x[0] else "_", zorder=3, hatch="//")
    ax2b.set_ylim(0, 1.35)
    ax2b.set_ylabel("Violation Rate  (DQN, 0–1)", color="#7f8c8d")
    ax2b.tick_params(axis="y", labelcolor="#7f8c8d")

    # combined legend
    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper left")

    # ══════════════════════════════════════════════════════════════════════════
    # ax3 — AR variability (std): Random vs Method
    # ══════════════════════════════════════════════════════════════════════════
    ax3.bar(x - bw/2, random_std, bw, color=pal_random, alpha=0.85, label="Random", zorder=3)
    for xi, (s, color) in enumerate(zip(method_std, pal_method_2)):
        ax3.bar(xi + bw/2, s, bw, color=color, alpha=0.85,
                label="PPO/DQN" if xi == 0 else "_", zorder=3)
    ax3.set_xticks(x); ax3.set_xticklabels(problem_names)
    ax3.set_ylabel("AR Standard Deviation (lower = more stable)")
    ax3.set_title("AR Variability: Random vs Method", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3, zorder=0)

    # ══════════════════════════════════════════════════════════════════════════
    # ax4 — ΔAR vs Random Baseline  +  gap to ILP
    # ══════════════════════════════════════════════════════════════════════════
    delta_random = [m - r for m, r in zip(method_ar, random_ar)]
    delta_ilp    = [m - i for m, i in zip(method_ar, ilp_ar)]

    bars_dr = ax4.bar(x - bw/2, delta_random, bw, alpha=0.85, zorder=3, label="vs Random")
    bars_di = ax4.bar(x + bw/2, delta_ilp,    bw, alpha=0.85, zorder=3, label="vs ILP")

    for bar, color in zip(bars_dr, pal_method_2):
        bar.set_facecolor(color)
    for bar in bars_di:
        bar.set_facecolor("#e74c3c")

    for xi, (dr, di) in enumerate(zip(delta_random, delta_ilp)):
        pad = 0.003
        ax4.text(xi - bw/2, dr + (pad if dr >= 0 else -pad*3),
                 f"{dr:+.3f}", ha="center", fontsize=8, fontweight="bold",
                 va="bottom" if dr >= 0 else "top", color=pal_method_2[xi])
        ax4.text(xi + bw/2, di + (pad if di >= 0 else -pad*3),
                 f"{di:+.3f}", ha="center", fontsize=8, fontweight="bold",
                 va="bottom" if di >= 0 else "top", color="#e74c3c")

    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xticks(x); ax4.set_xticklabels(problem_names)
    ax4.set_ylim(-0.25, 0.25)
    ax4.set_ylabel("ΔAR")
    ax4.set_title("AR Delta: Method vs Random / Method vs ILP", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.3, zorder=0)

    # ── Overall title & layout ─────────────────────────────────────────────────
    fig.suptitle("Combined Results: P3–P6 PPO Variants + DQN vs ILP Optimal",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def main():
    combined_data = []
    for path in PATH_LIST:
        data = read_csv(path)
        combined_data.append(data)
    
    ilp_data = read_json(ILP_DATA_PATH)
    
    # draw combined results
    plot_combined_results(combined_data, ilp_data, output_path=HERE / "combined_results.png")

    
    
if __name__ == "__main__":
    main()