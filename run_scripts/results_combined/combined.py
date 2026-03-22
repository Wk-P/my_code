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


def pick_metric(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        return float(value)
    return default
    

def plot_combined_results(data_list, ilp_data, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax1, ax2, ax3 = axes

    problem_names  = ["P3\nPPO", "P4\nPPO-Mask", "P5\nPPO-Lagrange", "P6\nPPO-Opt", "DQN"]
    pal_ilp    = "#e74c3c"
    pal_random = "#95a5a6"
    pal_method = ["#3498db" for _ in range(len(problem_names))]
    pal_method_2 = pal_method
    x  = np.arange(len(problem_names))
    bw = 0.26  # bar width

    # ── Parse each problem's three rows ────────────────────────────────────────
    ilp_ar, random_ar, method_ar   = [], [], []
    random_std, method_std         = [], []
    ilp_placed, random_placed, method_placed = [], [], []
    random_viol, method_viol       = [], []
    for data in data_list:
        def _row(kw):
            return next((r for r in data if kw.lower() in r["method"].lower()), {})
        ilp  = _row("ILP")
        rand = _row("Random")
        mth  = next((r for r in data
                     if "ILP" not in r["method"] and "Random" not in r["method"]), {})

        ilp_ar.append(pick_metric(ilp, "ar_mean"))
        random_ar.append(pick_metric(rand, "ar_mean"))
        method_ar.append(pick_metric(mth, "ar_mean"))
        random_std.append(pick_metric(rand, "ar_std"))
        method_std.append(pick_metric(mth, "ar_std"))
        ilp_placed.append(pick_metric(ilp, "placed_mean"))
        random_placed.append(pick_metric(rand, "placed_mean"))
        method_placed.append(pick_metric(mth, "placed_mean"))

        random_viol.append(pick_metric(rand, "viol_rate", "viol_per_ep", "violations"))
        method_viol.append(pick_metric(mth, "viol_rate", "viol_per_ep", "violations"))

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
    # ax2 — Constraint violation rate
    # ══════════════════════════════════════════════════════════════════════════
    for xi in range(len(problem_names)):
        ax2.bar(xi - bw/2, random_viol[xi], bw, color=pal_random, alpha=0.8,
                label="Random" if xi == 0 else "_", zorder=3)
        ax2.bar(xi + bw/2, method_viol[xi], bw, color=pal_method_2[xi], alpha=0.8,
                label="PPO / DQN" if xi == 0 else "_", zorder=3)
        ax2.text(xi + bw/2, method_viol[xi] + 0.015,
                 f"{method_viol[xi]:.2f}", ha="center", fontsize=8,
                 fontweight="bold", color=pal_method_2[xi])

    ax2.set_xticks(x); ax2.set_xticklabels(problem_names)
    ax2.set_ylabel("Violation Rate")
    ax2.set_title("Constraint Violation Rate", fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    # ══════════════════════════════════════════════════════════════════════════
    # ax3 — Placement completeness
    # ══════════════════════════════════════════════════════════════════════════
    ax3.bar(x - bw, ilp_placed, bw, color=pal_ilp, alpha=0.85, label="ILP (Optimal)", zorder=3)
    ax3.bar(x, random_placed, bw, color=pal_random, alpha=0.85, label="Random", zorder=3)
    for xi, (placed, color) in enumerate(zip(method_placed, pal_method_2)):
        ax3.bar(xi + bw, placed, bw, color=color, alpha=0.85,
                label="PPO / DQN" if xi == 0 else "_", zorder=3)
        ax3.text(xi + bw, placed + 0.12, f"{placed:.1f}",
                 ha="center", fontsize=8, fontweight="bold", color=color)
    ax3.set_xticks(x); ax3.set_xticklabels(problem_names)
    ax3.set_ylabel("Services Placed")
    ax3.set_title("Placement Completeness", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3, zorder=0)

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
        if not path.exists():
            raise FileNotFoundError(f"Missing combined input CSV: {path}")
        data = read_csv(path)
        combined_data.append(data)
    
    ilp_data = read_json(ILP_DATA_PATH)
    
    # draw combined results
    plot_combined_results(combined_data, ilp_data, output_path=HERE / "combined_results.png")

    
    
if __name__ == "__main__":
    main()