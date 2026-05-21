"""
plot_metrics.py — Four focused metric figures for the ECU placement study.

Figure 1  (fig1_ar_comparison):    AR per method, all 3 scenarios side-by-side.
                                   Bars color-coded by constraint strategy;
                                   violation count annotated directly on the bar.
Figure 2  (fig2_violations):       Total constraint violations per method, stacked
                                   cap + conflict, y-axis clipped for readability.
Figure 3  (fig3_tradeoff_scatter): AR (% of ILP) vs violations per episode —
                                   the safety–performance frontier.
Figure 4  (fig4_full_placement):   Line chart — average number of fully-placed
                                   test episodes (services_placed == M) per method
                                   across scenarios LT / EQ / GT.
"""

import csv
import datetime
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

ROOT         = Path(__file__).parent.parent
CSV_PATH     = ROOT / "logs/multi_seed_results/aggregate_summary.csv"
RESULTS_ROOT = ROOT / "results"

# Map full group name → short key
_GROUP_SHORTS = {
    "ecu_lt_svc_p": "lt",
    "ecu_eq_svc_p": "eq",
    "ecu_gt_svc_p": "gt",
}

# ILP cache location per group (use problem4 as canonical)
_ILP_PKG = {
    "lt": ROOT / "ecu_lt_svc_p" / "problem4_ppo_mask" / "results" / "ilp_cache.json",
    "eq": ROOT / "ecu_eq_svc_p" / "problem4_ppo_mask" / "results" / "ilp_cache.json",
    "gt": ROOT / "ecu_gt_svc_p" / "problem4_ppo_mask" / "results" / "ilp_cache.json",
}

# Fallback window (minutes) for legacy seed dirs that have no .run_id file.
# Seeds whose test_results.csv mtime falls within this window of the most-recent
# seed are treated as one batch. New runs always write .run_id, so this fallback
# only matters for results generated before run_all_parallel.py was adopted.
_FALLBACK_WINDOW_MINUTES = 30

# Output root — all figures from one run share a single dated subfolder
FIGURES_ROOT = ROOT / "figures"
_RUN_DIR: Path | None = None  # set once in main() before any save_fig call

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
    ("problem4_ppo_mask",       "Maskable PPO"),
    ("problem5_ppo_lagrangian", "Lagrangian PPO"),
    ("problem6_ppo_opt",        "PPO & Optimal\nAlgorithm"),
    ("problem_dqn",             "DQN"),
    ("problem_ddqn",            "DDQN"),
    ("problem7_ppo_seq",        "Seq PPO\n(P7)"),
]

ILP_COLOR = "#d62728"

# Per-method bar style (uniform for bar charts)
BAR_COLOR = "#4878d0"
METHOD_STYLES = {
    "problem3_ppo":            {"color": BAR_COLOR, "hatch": None},
    "problem4_ppo_mask":       {"color": BAR_COLOR, "hatch": None},
    "problem5_ppo_lagrangian": {"color": BAR_COLOR, "hatch": None},
    "problem6_ppo_opt":        {"color": BAR_COLOR, "hatch": None},
    "problem_dqn":             {"color": BAR_COLOR, "hatch": None},
    "problem_ddqn":            {"color": BAR_COLOR, "hatch": None},
}

# Per-method distinct colors for scatter plot
SCATTER_COLORS = {
    "problem4_ppo_mask":       "#2ca02c",
    "problem5_ppo_lagrangian": "#1f77b4",
    "problem6_ppo_opt":        "#17becf",
    "problem_dqn":             "#ff7f0e",
    "problem_ddqn":            "#9467bd",
    "problem7_ppo_seq":        "#e377c2",
}

LEGEND_HANDLES = [
    mpatches.Patch(facecolor=BAR_COLOR, edgecolor="black", label="Maskable PPO"),
    mpatches.Patch(facecolor=BAR_COLOR, edgecolor="black", label="Lagrangian PPO"),
    mpatches.Patch(facecolor=BAR_COLOR, edgecolor="black", label="PPO & Optimal Algorithm"),
    mpatches.Patch(facecolor=BAR_COLOR, edgecolor="black", label="DQN"),
    mpatches.Patch(facecolor=BAR_COLOR, edgecolor="black", label="DDQN"),
    mpatches.Patch(facecolor=BAR_COLOR, edgecolor="black", label="Seq PPO (P7)"),
]

SCATTER_LEGEND_HANDLES = [
    mpatches.Patch(facecolor=SCATTER_COLORS["problem4_ppo_mask"],       edgecolor="black", label="Maskable PPO"),
    mpatches.Patch(facecolor=SCATTER_COLORS["problem5_ppo_lagrangian"], edgecolor="black", label="Lagrangian PPO"),
    mpatches.Patch(facecolor=SCATTER_COLORS["problem6_ppo_opt"],        edgecolor="black", label="PPO & Optimal Algorithm"),
    mpatches.Patch(facecolor=SCATTER_COLORS["problem_dqn"],             edgecolor="black", label="DQN"),
    mpatches.Patch(facecolor=SCATTER_COLORS["problem_ddqn"],            edgecolor="black", label="DDQN"),
    mpatches.Patch(facecolor=SCATTER_COLORS["problem7_ppo_seq"],        edgecolor="black", label="Seq PPO (P7)"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

EMPTY_VALS = {
    "ar": 0.0, "ar_std": 0.0, "placed": 0.0,
    "viol_rate": 0.0, "cap": 0.0, "conflict": 0.0,
}


def _latest_seed_dirs(group_dir: Path) -> list:
    """Return seed dirs that belong to the latest experiment run.

    Priority 1 — .run_id files present (written by run_experiment.py):
        Group seed dirs by their run_id tag; return dirs from the run_id
        whose newest seed has the highest mtime.

    Priority 2 — legacy fallback (no .run_id files anywhere in this group):
        Return seed dirs whose test_results.csv mtime is within
        _FALLBACK_WINDOW_MINUTES of the most-recently-modified seed.
        Tight window (30 min) avoids mixing results from different sessions.
    """
    tagged: dict[str, list] = {}   # run_id  → [(mtime, dir), ...]
    untagged: list = []             # (mtime, dir) without .run_id

    for d in sorted(group_dir.glob("seed_*")):
        csv_file   = d / "test_results.csv"
        run_id_file = d / ".run_id"
        if not csv_file.exists():
            continue
        mtime = csv_file.stat().st_mtime
        if run_id_file.exists():
            rid = run_id_file.read_text().strip()
            tagged.setdefault(rid, []).append((mtime, d))
        else:
            untagged.append((mtime, d))

    if tagged:
        # Pick the run_id whose most-recent seed has the highest mtime
        latest_rid = max(tagged, key=lambda rid: max(m for m, _ in tagged[rid]))
        return [d for _, d in tagged[latest_rid]]

    # Fallback for legacy data: tight time-window around the most-recent seed
    if not untagged:
        return []
    max_mtime = max(m for m, _ in untagged)
    cutoff = max_mtime - _FALLBACK_WINDOW_MINUTES * 60
    return [d for m, d in untagged if m >= cutoff]


def _load_ilp_feasibility() -> dict:
    """Return {group_short: feasibility_rate} from ILP cache files."""
    rates = {}
    for group_short, cache_path in _ILP_PKG.items():
        if not cache_path.exists():
            rates[group_short] = 1.0
            continue
        with open(cache_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        if not results:
            rates[group_short] = 1.0
            continue
        feasible = sum(1 for r in results if r.get("status") == "Optimal")
        rates[group_short] = feasible / len(results)
    return rates


def _load_success_rates() -> dict:
    """Return {group_short: {prob_key: mean_success_rate}} from latest-batch test_results.csv."""
    result = {}
    for group_short, _ in _FIG4_GROUPS:
        group_dir = RESULTS_ROOT / group_short
        if not group_dir.exists():
            result[group_short] = {}
            continue
        per_prob: dict = defaultdict(list)
        for seed_dir in _latest_seed_dirs(group_dir):
            csv_file = seed_dir / "test_results.csv"
            if not csv_file.exists():
                continue
            with open(csv_file, newline="") as f:
                for row in csv.DictReader(f):
                    raw = row["model"]
                    if raw not in _MODEL_MAP:
                        continue
                    prob_key, _ = _MODEL_MAP[raw]
                    no_viol = (row["episode_has_cap_violation"] == "0" and
                               row["episode_has_conflict_violation"] == "0")
                    success = int(int(row["services_placed"]) == int(row["M"]) and no_viol)
                    per_prob[prob_key].append(success)
        result[group_short] = {pk: float(np.mean(v)) for pk, v in per_prob.items()}
    return result


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
    """Return (methods, ilp_ar, ilp_placed) where methods = [(prob_key, label, vals), ...]."""
    ilp_ar = None
    ilp_placed = None
    for (g, p, m), v in data.items():
        if g == group and "ILP" in m:
            ilp_ar = v["ar"]
            ilp_placed = v["placed"]
            break

    methods = []
    for prob, label in PROBLEM_ORDER:
        found = None
        for (g, p, m), v in data.items():
            if g == group and p == prob and "ILP" not in m:
                found = v
                break
        methods.append((prob, label, found or EMPTY_VALS.copy()))

    return methods, ilp_ar, ilp_placed


def save_fig(fig, stem: str):
    assert _RUN_DIR is not None, "call set_run_dir() before save_fig()"
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = _RUN_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")


# ── Figure 1: AR comparison (constrained methods only) ───────────────────────

def plot_ar_comparison(data: dict):
    """AR comparison for constrained methods P4/P5/P6/DQN/DDQN.
    P3 is excluded — its results are available in aggregate_summary.csv only."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "AR Comparison — Constrained Methods vs ILP Optimum",
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
        group_short = _GROUP_SHORTS.get(group, "")
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
            ax.bar(xs[i], ar_disp, width=w,
                   color=style["color"], hatch=style["hatch"],
                   edgecolor="black", linewidth=0.6)

            label_y = ar_disp + YMAX * 0.01
            ax.text(xs[i], label_y, f"{ar_true:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")



        ax.set_ylim(0, YMAX)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in PROBLEM_ORDER], fontsize=8.5, rotation=15, ha="right")
        ax.set_xlabel("RL Models", fontsize=9)
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
        methods, _, ilp_placed = get_scenario(group, data)

        caps  = np.array([v["cap"]      for _, _, v in methods])
        confs = np.array([v["conflict"] for _, _, v in methods])
        tots  = caps + confs

        ymax_vis = max(float(np.max(tots)) * 1.35, 5.0)

        ax.bar(xs, caps,  width=w, color=CAP_COLOR,  edgecolor="black",
               linewidth=0.5, label="Capacity violations")
        ax.bar(xs, confs, width=w, bottom=caps, color=CONF_COLOR, edgecolor="black",
               linewidth=0.5, label="Conflict violations")

        placed_total = ilp_placed if ilp_placed else 1.0
        for i, (tot, (_, _, v)) in enumerate(zip(tots, methods)):
            label_y = tot + ymax_vis * 0.01
            if tot > 0:
                ax.text(xs[i], label_y + ymax_vis * 0.04, f"{tot:.0f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            else:
                label_y = ymax_vis * 0.01
            placed_txt = f"{v['placed']:.1f}/{placed_total:.0f}"
            ax.text(xs[i], label_y, placed_txt,
                    ha="center", va="bottom", fontsize=7, color="#555555")

        ax.set_ylim(0, ymax_vis)
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in PROBLEM_ORDER], fontsize=8.5, rotation=15, ha="right")
        ax.set_xlabel("RL Models", fontsize=9)
        ax.set_title(scene_long, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Total violations (40-episode test set)", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if col == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
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
        methods, ilp_ar, _ = get_scenario(group, data)

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
            color = SCATTER_COLORS.get(prob, "#1f77b4")
            x, y  = xvals[i], yvals[i]
            lbl   = SHORT_LABELS.get(prob, "?")

            ax.scatter([x], [y], s=110, color=color,
                       edgecolors="black", linewidth=0.6, zorder=5)
            # offset label to avoid overlap
            offset = (6, 5) if x < x_max * 0.7 else (-28, 5)
            ax.annotate(
                lbl, (x, y),
                textcoords="offset points", xytext=offset,
                fontsize=9, fontweight="bold", color=color,
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

    fig.legend(handles=SCATTER_LEGEND_HANDLES, loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    save_fig(fig, "fig3_tradeoff_scatter")
    plt.close()


# ── Figure 4: Placed SVCs per test scenario (3 subplots) ─────────────────────

# Map from test_results.csv model name → (PROBLEM_ORDER key, display label)
_MODEL_MAP = {
    "P4_MaskPPO":   ("problem4_ppo_mask",        "Maskable PPO"),
    "P5_LagPPO":    ("problem5_ppo_lagrangian",   "Lagrangian PPO"),
    "P6_RepairPPO": ("problem6_ppo_opt",           "PPO & Optimal\nAlgorithm"),
    "DQN":          ("problem_dqn",                "DQN"),
    "DDQN":         ("problem_ddqn",               "DDQN"),
    "P7_SeqPPO":    ("problem7_ppo_seq",           "Seq PPO (P7)"),
}

_FIG4_GROUPS = [
    ("lt", "LT  (N=10 ECUs < M=15 SVCs)\nResource-scarce"),
    ("eq", "EQ  (N=M=10)\nBalanced"),
    ("gt", "GT  (N=15 ECUs > M=10 SVCs)\nResource-abundant"),
]

_LINE_COLORS = {
    "problem4_ppo_mask":       SCATTER_COLORS["problem4_ppo_mask"],
    "problem5_ppo_lagrangian": SCATTER_COLORS["problem5_ppo_lagrangian"],
    "problem6_ppo_opt":        SCATTER_COLORS["problem6_ppo_opt"],
    "problem_dqn":             SCATTER_COLORS["problem_dqn"],
    "problem_ddqn":            SCATTER_COLORS["problem_ddqn"],
    "problem7_ppo_seq":        SCATTER_COLORS["problem7_ppo_seq"],
}


def _load_placed_per_scenario() -> dict:
    """Return {group_short: {prob_key: array[n_seeds, n_scenarios]}} of services_placed.
    Only uses the most recently modified batch of seeds per group."""
    result: dict = {}
    for group_short, _ in _FIG4_GROUPS:
        group_dir = RESULTS_ROOT / group_short
        if not group_dir.exists():
            result[group_short] = {}
            continue
        # {prob_key: [seed0_array, seed1_array, ...]}
        per_prob: dict = defaultdict(list)
        for seed_dir in _latest_seed_dirs(group_dir):
            csv_file = seed_dir / "test_results.csv"
            if not csv_file.exists():
                continue
            # {prob_key: {scenario_idx: placed}}
            seed_data: dict = defaultdict(dict)
            with open(csv_file, newline="") as f:
                for row in csv.DictReader(f):
                    raw = row["model"]
                    if raw not in _MODEL_MAP:
                        continue
                    prob_key, _ = _MODEL_MAP[raw]
                    idx     = int(row["scenario_idx"])
                    no_viol    = (row["episode_has_cap_violation"] == "0" and
                                  row["episode_has_conflict_violation"] == "0")
                    full_valid = int(int(row["services_placed"]) == int(row["M"]) and no_viol)
                    seed_data[prob_key][idx] = full_valid
            for prob_key, idx_map in seed_data.items():
                n = max(idx_map.keys()) + 1
                arr = np.array([idx_map.get(i, 0) for i in range(n)], dtype=float)
                per_prob[prob_key].append(arr)
        result[group_short] = {pk: np.array(arrs) for pk, arrs in per_prob.items()}
    return result


def plot_full_placement():
    """Bar chart: overall full valid placement rate per method per group.
    Uses only the most recent training batch; ILP reference from actual cache feasibility."""
    placed_data = _load_placed_per_scenario()
    ilp_rates   = _load_ilp_feasibility()

    # Count actual seeds used per group for title
    seed_counts = {}
    for group_short, _ in _FIG4_GROUPS:
        group_dir = RESULTS_ROOT / group_short
        seed_counts[group_short] = len(_latest_seed_dirs(group_dir)) if group_dir.exists() else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    fig.suptitle(
        "Full Valid Placement Rate  (latest training batch only)\n"
        "Rate = fraction of (scenario, seed) pairs where all M SVCs placed with zero violations",
        fontsize=11, fontweight="bold",
    )

    bar_width = 0.55
    for col, (group_short, scene_long) in enumerate(_FIG4_GROUPS):
        ax = axes[col]
        prob_arrays = placed_data.get(group_short, {})
        ilp_rate    = ilp_rates.get(group_short, 1.0)
        n_used      = seed_counts.get(group_short, 0)

        labels, means, sems, colors = [], [], [], []
        for prob_key, label in PROBLEM_ORDER:
            arrs = prob_arrays.get(prob_key)
            if arrs is None or len(arrs) == 0:
                continue
            flat        = arrs.flatten()
            mean        = flat.mean()
            scene_means = arrs.mean(axis=0)
            sem         = scene_means.std() / np.sqrt(len(scene_means))
            labels.append(label.replace("\n", " "))
            means.append(mean)
            sems.append(sem)
            colors.append(_LINE_COLORS[prob_key])

        xs = np.arange(len(labels))
        ax.bar(xs, means, width=bar_width, color=colors, edgecolor="black",
               linewidth=0.7, zorder=3)
        ax.errorbar(xs, means, yerr=sems, fmt="none", color="black",
                    capsize=4, linewidth=1.2, zorder=4)

        for x, m in zip(xs, means):
            ax.text(x, m + 0.02, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

        ax.axhline(ilp_rate, color=ILP_COLOR, linestyle="--", linewidth=1.5,
                   label=f"ILP  {ilp_rate:.0%} feasible")
        ax.set_ylim(0, 1.18)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Full valid placement rate", fontsize=9)
        title_suffix = f"\n(n={n_used} seed{'s' if n_used != 1 else ''})"
        ax.set_title(scene_long + title_suffix, fontsize=9.5, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        if col == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    save_fig(fig, "fig4_full_placement")
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────

def _collect_training_curves():
    """Copy training_curve.png from latest seed dirs into _RUN_DIR/training_curves/."""
    import shutil
    assert _RUN_DIR is not None
    out_dir = _RUN_DIR / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    for group_short, _ in _FIG4_GROUPS:
        group_dir = RESULTS_ROOT / group_short
        if not group_dir.exists():
            continue
        for seed_dir in _latest_seed_dirs(group_dir):
            src = seed_dir / "training_curve.png"
            if src.exists():
                dst = out_dir / f"{group_short}_{seed_dir.name}_training_curve.png"
                shutil.copy2(src, dst)
                print(f"Copied: {dst.name}")


def main():
    global _RUN_DIR
    import os
    data     = load_csv(CSV_PATH)
    dt       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag      = os.urandom(4).hex()
    _RUN_DIR = FIGURES_ROOT / f"{dt}_{tag}"

    plot_ar_comparison(data)
    plot_violations(data)
    plot_tradeoff(data)
    plot_full_placement()
    _collect_training_curves()


if __name__ == "__main__":
    main()
