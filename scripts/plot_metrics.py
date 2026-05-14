"""
plot_metrics.py — Three focused metric figures for the ECU placement study.

Figure 1  (fig1_ar_comparison):    AR per method, all 3 scenarios side-by-side.
                                   Bars color-coded by constraint strategy;
                                   violation count annotated directly on the bar.
Figure 2  (fig2_violations):       Total constraint violations per method, stacked
                                   cap + conflict, y-axis clipped for readability.
Figure 3  (fig3_tradeoff_scatter): AR (% of ILP) vs violations per episode —
                                   the safety–performance frontier.
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

ROOT     = Path(__file__).parent.parent
CSV_PATH = ROOT / "logs/multi_seed_results/aggregate_summary.csv"

# Output root — each figure gets its own subfolder
FIGURES_ROOT = ROOT / "figures"
FIG_DIRS = {
    "fig1_ar_comparison":  FIGURES_ROOT / "fig1_ar_comparison",
    "fig2_violations":     FIGURES_ROOT / "fig2_violations",
    "fig3_tradeoff_scatter": FIGURES_ROOT / "fig3_tradeoff",
}

# ── scenario definitions ──────────────────────────────────────────────────────
ROW_GROUPS = ["ecu_lt_svc_p", "ecu_eq_svc_p", "ecu_gt_svc_p"]
SCENE_LONG = [
    "LT  (N=10 ECUs < M=15 SVCs)\nResource-scarce",
    "EQ  (N=M=10)\nBalanced",
    "GT  (N=15 ECUs > M=10 SVCs)\nResource-abundant",
]
SCENE_SHORT = ["LT", "EQ", "GT"]

# P3 is excluded from all charts — its results are CSV-only (alongside ILP).
# All figures use PROBLEM_ORDER which contains only the constrained methods.
PROBLEM_ORDER = [
    ("problem4_ppo_mask",       "Mask\n(P4)"),
    ("problem5_ppo_lagrangian", "Lagrange\n(P5)"),
    ("problem6_ppo_opt",        "Repair\n(P6)"),
    ("problem_dqn",             "DQN"),
    ("problem_ddqn",            "DDQN"),
]

ILP_COLOR = "#d62728"

# Per-method bar style
METHOD_STYLES = {
    "problem3_ppo":            {"color": "#d62728", "hatch": "////"},
    "problem4_ppo_mask":       {"color": "#2ca02c", "hatch": None},
    "problem5_ppo_lagrangian": {"color": "#1f77b4", "hatch": None},
    "problem6_ppo_opt":        {"color": "#17becf", "hatch": None},
    "problem_dqn":             {"color": "#ff7f0e", "hatch": "..."},
    "problem_ddqn":            {"color": "#ff7f0e", "hatch": "xxx"},
}

LEGEND_HANDLES = [
    mpatches.Patch(facecolor="#2ca02c", edgecolor="black",
                   label="P4 — Hard mask (zero violations guaranteed)"),
    mpatches.Patch(facecolor="#1f77b4", edgecolor="black",
                   label="P5 — Lagrangian / soft constraint"),
    mpatches.Patch(facecolor="#17becf", edgecolor="black",
                   label="P6 — Repair heuristic"),
    mpatches.Patch(facecolor="#ff7f0e", hatch="...", edgecolor="black",
                   label="DQN"),
    mpatches.Patch(facecolor="#ff7f0e", hatch="xxx", edgecolor="black",
                   label="DDQN"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

EMPTY_VALS = {
    "ar": 0.0, "ar_std": 0.0, "placed": 0.0,
    "viol_rate": 0.0, "cap": 0.0, "conflict": 0.0,
}


def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def load_csv(path: Path) -> dict:
    data: dict = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["group"], row["problem"], row["method"])
            data[key] = {
                "ar":        safe_float(row["ar_mean_avg"]),
                "ar_std":    safe_float(row["ar_mean_std"]),
                "placed":    safe_float(row["placed_mean_avg"]),
                "viol_rate": safe_float(row.get("viol_rate_avg", 0)),
                "cap":       safe_float(row["cap_viol_avg"]),
                "conflict":  safe_float(row["conflict_viol_avg"]),
            }
    return data


def get_scenario(group: str, data: dict):
    """Return (methods, ilp_ar) where methods = [(prob_key, label, vals), ...]."""
    ilp_ar = None
    for (g, p, m), v in data.items():
        if g == group and "ILP" in m:
            ilp_ar = v["ar"]
            break

    methods = []
    for prob, label in PROBLEM_ORDER:
        found = None
        for (g, p, m), v in data.items():
            if g == group and p == prob and "ILP" not in m:
                found = v
                break
        methods.append((prob, label, found or EMPTY_VALS.copy()))

    return methods, ilp_ar


def save_fig(fig, stem: str):
    out_dir = FIG_DIRS.get(stem, FIGURES_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")


# ── Figure 1: AR comparison (constrained methods only) ───────────────────────

def plot_ar_comparison(data: dict):
    """AR comparison for constrained methods P4/P5/P6/DQN/DDQN.
    P3 is excluded — its results are available in aggregate_summary.csv only."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "AR Comparison — Constrained Methods vs ILP Optimum\n"
        "Violation count annotated above bar  (✗N = cap + conflict violations over 40 test episodes)",
        fontsize=12, fontweight="bold",
    )

    YMAX = 1.08
    CLIP = 1.00

    xs = np.arange(len(PROBLEM_ORDER))
    w  = 0.6

    for col, (group, scene_long, short) in enumerate(
        zip(ROW_GROUPS, SCENE_LONG, SCENE_SHORT)
    ):
        ax = axes[col]
        ilp_ar = None
        for (g, p, m), v in data.items():
            if g == group and "ILP" in m:
                ilp_ar = v["ar"]
                break

        methods_main = []
        for prob, label in PROBLEM_ORDER:
            found = None
            for (g, p, m), v in data.items():
                if g == group and p == prob and "ILP" not in m:
                    found = v
                    break
            methods_main.append((prob, label, found or EMPTY_VALS.copy()))

        if ilp_ar is not None:
            ax.axhline(ilp_ar, color=ILP_COLOR, linestyle="--", linewidth=1.8,
                       label=f"ILP = {ilp_ar:.3f}")
            ax.legend(fontsize=8, loc="lower right")

        for i, (prob, _label, v) in enumerate(methods_main):
            style   = METHOD_STYLES.get(prob, {"color": "#1f77b4", "hatch": None})
            ar_true = v["ar"]
            ar_disp = min(ar_true, CLIP)
            total_v = v["cap"] + v["conflict"]

            ax.bar(xs[i], ar_disp, width=w,
                   color=style["color"], hatch=style["hatch"],
                   edgecolor="black", linewidth=0.6)

            label_y = ar_disp + YMAX * 0.01
            ax.text(xs[i], label_y, f"{ar_true:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

            if total_v > 0:
                ax.text(xs[i], label_y + YMAX * 0.055, f"✗{total_v:.0f}",
                        ha="center", va="bottom", fontsize=8,
                        color="#d62728", fontweight="bold")

        ax.set_ylim(0, YMAX)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in PROBLEM_ORDER], fontsize=9)
        ax.set_title(scene_long, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Average Resource Utilization (AR)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    save_fig(fig, "fig1_ar_comparison")
    plt.close()


# ── Figure 2: Violations ──────────────────────────────────────────────────────

def plot_violations(data: dict):
    """Stacked violation bars for constrained methods (P3 excluded — CSV only)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "Constraint Violations by Method  —  Stacked: Capacity (light) + Conflict (dark)\n"
        "P3 (unconstrained) omitted from chart; its violation data is in aggregate_summary.csv",
        fontsize=12, fontweight="bold",
    )

    CAP_COLOR  = "#ff9f4a"
    CONF_COLOR = "#c0392b"

    xs = np.arange(len(PROBLEM_ORDER))
    w  = 0.6

    for col, (group, scene_long, short) in enumerate(
        zip(ROW_GROUPS, SCENE_LONG, SCENE_SHORT)
    ):
        ax = axes[col]
        methods, _ = get_scenario(group, data)

        caps  = np.array([v["cap"]      for _, _, v in methods])
        confs = np.array([v["conflict"] for _, _, v in methods])
        tots  = caps + confs

        ymax_vis = max(float(np.max(tots)) * 1.35, 5.0)

        ax.bar(xs, caps,  width=w, color=CAP_COLOR,  edgecolor="black",
               linewidth=0.5, label="Capacity violations")
        ax.bar(xs, confs, width=w, bottom=caps, color=CONF_COLOR, edgecolor="black",
               linewidth=0.5, label="Conflict violations")

        for i, tot in enumerate(tots):
            if tot == 0:
                continue
            ax.text(xs[i], tot + ymax_vis * 0.01, f"{tot:.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylim(0, ymax_vis)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in PROBLEM_ORDER], fontsize=8.5)
        ax.set_title(scene_long, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Total violations (40-episode test set)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if col == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    save_fig(fig, "fig2_violations")
    plt.close()


# ── Figure 3: Safety–Performance Tradeoff ────────────────────────────────────

def plot_tradeoff(data: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "Safety–Performance Frontier:  AR (% of ILP)  vs  Violations per Episode\n"
        "Green zone = practically feasible ( < 0.5 violations/episode );  "
        "red zone = unacceptable constraint violations",
        fontsize=12, fontweight="bold",
    )

    FEASIBLE_THR = 0.5   # violations per episode
    N_TEST_EPS   = 40

    SHORT_LABELS = {
        "problem4_ppo_mask":       "P4",
        "problem5_ppo_lagrangian": "P5",
        "problem6_ppo_opt":        "P6",
        "problem_dqn":             "DQN",
        "problem_ddqn":            "DDQN",
    }

    for col, (group, scene_long, short) in enumerate(
        zip(ROW_GROUPS, SCENE_LONG, SCENE_SHORT)
    ):
        ax = axes[col]
        methods, ilp_ar = get_scenario(group, data)

        # x = total violations / 40 test episodes
        # y = AR / ilp_ar  (1.0 = ILP optimum)
        xvals = [(v["cap"] + v["conflict"]) / N_TEST_EPS for _, _, v in methods]
        yvals = [v["ar"] / ilp_ar if (ilp_ar and ilp_ar > 0) else v["ar"]
                 for _, _, v in methods]

        x_max = max(max(xvals) * 1.15, FEASIBLE_THR * 4)

        # shade zones
        ax.axvspan(0,            FEASIBLE_THR, color="#d4edda", alpha=0.55)
        ax.axvspan(FEASIBLE_THR, x_max,        color="#f8d7da", alpha=0.45)
        ax.axvline(FEASIBLE_THR, color="green", linestyle=":", linewidth=1.2)

        # ILP gold star at (0, 1.0)
        ax.scatter([0], [1.0], marker="*", s=260, color="gold",
                   edgecolors="#8b6914", linewidth=0.8, zorder=6,
                   label=f"ILP  AR={ilp_ar:.3f}")

        # plot each method
        for i, (prob, _label, v) in enumerate(methods):
            style = METHOD_STYLES.get(prob, {"color": "#1f77b4", "hatch": None})
            x, y  = xvals[i], yvals[i]
            lbl   = SHORT_LABELS.get(prob, "?")

            ax.scatter([x], [y], s=110, color=style["color"],
                       edgecolors="black", linewidth=0.6, zorder=5)
            # offset label to avoid overlap
            offset = (6, 5) if x < x_max * 0.7 else (-28, 5)
            ax.annotate(
                lbl, (x, y),
                textcoords="offset points", xytext=offset,
                fontsize=9, fontweight="bold", color=style["color"],
            )

        ax.set_xlim(0, x_max)
        y_all  = yvals + [1.0]
        y_lo   = min(y_all) - 0.015
        y_hi   = max(y_all) + 0.03
        ax.set_ylim(y_lo, y_hi)

        ax.set_xlabel("Violations per episode  (cap + conflict) / 40", fontsize=9)
        ax.set_ylabel("AR / ILP optimum", fontsize=9)
        ax.set_title(scene_long, fontsize=9.5, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")

        # zone labels
        ax.text(FEASIBLE_THR * 0.1, y_lo + (y_hi - y_lo) * 0.04,
                "feasible\nzone", fontsize=7.5, color="green", alpha=0.8)
        ax.text(FEASIBLE_THR * 1.1, y_lo + (y_hi - y_lo) * 0.04,
                "infeasible\nzone", fontsize=7.5, color="#c0392b", alpha=0.8)

    fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    save_fig(fig, "fig3_tradeoff_scatter")
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    data = load_csv(CSV_PATH)
    plot_ar_comparison(data)
    plot_violations(data)
    plot_tradeoff(data)


if __name__ == "__main__":
    main()
