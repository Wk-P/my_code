"""
Cross-seed summary figures for one experiment.

Functions:
    plot_experiment_summary(groups_agg, ilp_data, out_dir)
        Generates 4 figures (AR / Violations / Tradeoff / Success+Placement)
        and saves them as PNG into out_dir.

    load_groups_agg(group_seed_dirs)
        Reads test_results_agg.csv from each seed dir and returns a
        {group: {model: aggregated_dict}} mapping ready for plotting.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = ["eq", "lt", "gt"]
GL     = {"eq": "EQ  (N=M=10)", "lt": "LT  (N<M)", "gt": "GT  (N>M)"}
MODELS = ["P3_PPO", "P4_MaskPPO", "P5_LagPPO", "P6_RepairPPO", "DQN", "DDQN", "P7_SeqPPO"]
MS     = {
    "P3_PPO":       "P3",
    "P4_MaskPPO":   "P4\nMask",
    "P5_LagPPO":    "P5\nLag",
    "P6_RepairPPO": "P6\nRepair",
    "DQN":          "DQN",
    "DDQN":         "DDQN",
    "P7_SeqPPO":    "P7\nSeq",
}
MODEL_COLORS = {
    "P3_PPO":       "#4c78a8",
    "P4_MaskPPO":   "#f58518",
    "P5_LagPPO":    "#e45756",
    "P6_RepairPPO": "#72b7b2",
    "DQN":          "#54a24b",
    "DDQN":         "#b279a2",
    "P7_SeqPPO":    "#e377c2",
}
_DEFAULT_COLOR = "#aaaaaa"
ILP_COLOR      = "#d62728"


# ── data loading ──────────────────────────────────────────────────────────────

def _read_agg_csv(path: Path) -> dict[str, dict]:
    """Return {model: row_dict} from one test_results_agg.csv."""
    result = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                result[row["model"]] = {k: float(v) for k, v in row.items()
                                        if k != "model"}
    except Exception:
        pass
    return result


def load_groups_agg(
    group_seed_dirs: dict[str, list[Path]],
    agg_csv_name: str = "test_results_agg.csv",
) -> dict[str, dict[str, dict]]:
    """
    group_seed_dirs: {group: [seed_dir, ...]}
      Each seed_dir should contain test_results_agg.csv (legacy path)
      OR test_results/data/test_results_agg.csv (new path).

    Returns: {group: {model: {ar, ar_std, success, conf_viol, cap_viol, placed, M}}}
    """
    result: dict[str, dict[str, dict]] = {}

    for group, seed_dirs in group_seed_dirs.items():
        per_model: dict[str, list[dict]] = defaultdict(list)

        for sd in seed_dirs:
            # Try new path first, fall back to legacy
            for candidate in [
                sd / "test_results" / "data" / agg_csv_name,
                sd / agg_csv_name,
            ]:
                if candidate.exists():
                    for model, row in _read_agg_csv(candidate).items():
                        per_model[model].append(row)
                    break

        group_agg: dict[str, dict] = {}
        for model in MODELS:
            rows = per_model.get(model, [])
            if not rows:
                continue
            group_agg[model] = {
                "ar":        float(np.mean([r["ar_mean"]        for r in rows])),
                "ar_std":    float(np.std( [r["ar_mean"]        for r in rows])),
                "success":   float(np.mean([r["success_rate"]   for r in rows])),
                "failure":   float(np.mean([r["failure_rate"]   for r in rows])),
                "cap_viol":  float(np.mean([r["cap_viol_rate"]  for r in rows])),
                "conf_viol": float(np.mean([r["conf_viol_rate"] for r in rows])),
                "placed":    float(np.mean([r["placed_mean"]    for r in rows])),
                "M":         int(rows[0]["M"]),
                "n_seeds":   len(rows),
            }
        if group_agg:
            result[group] = group_agg

    return result


# ── shared helpers ────────────────────────────────────────────────────────────

def _present_groups(groups_agg: dict) -> list[str]:
    return [g for g in GROUPS if g in groups_agg]


def _models_in_group(group_agg: dict) -> list[str]:
    return [m for m in MODELS if m in group_agg]


def _fig3cols(groups_agg: dict, figw_per_col: float = 5.5, figh: float = 5.5):
    pg = _present_groups(groups_agg)
    ncols = len(pg)
    fig, axes = plt.subplots(1, ncols, figsize=(figw_per_col * ncols, figh),
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    return fig, axes, pg


# ── Figure 1: AR comparison ───────────────────────────────────────────────────

def fig_ar_comparison(
    groups_agg: dict[str, dict[str, dict]],
    ilp_data:   dict[str, float | None],
    out_dir:    Path,
) -> Path:
    fig, axes, pg = _fig3cols(groups_agg)
    fig.suptitle("AR Comparison — all models vs ILP optimum",
                 fontsize=12, fontweight="bold")

    for ax, g in zip(axes, pg):
        gd     = groups_agg[g]
        models = _models_in_group(gd)
        x      = np.arange(len(models))
        ilp    = ilp_data.get(g)

        if ilp is not None:
            ax.axhline(ilp, color=ILP_COLOR, linestyle="--", linewidth=1.8,
                       label=f"ILP = {ilp:.3f}")

        ar_vals = [min(gd[m]["ar"], 1.05) for m in models]
        ar_stds = [gd[m]["ar_std"]        for m in models]
        bars = ax.bar(x, ar_vals,
                      color=[MODEL_COLORS.get(m, _DEFAULT_COLOR) for m in models],
                      alpha=0.82, yerr=ar_stds, capsize=4,
                      error_kw={"linewidth": 1}, edgecolor="white", linewidth=0.4)

        for bar, m, vd in zip(bars, models, ar_vals):
            raw = gd[m]["ar"]
            lbl = f"{raw:.3f}" + (" ▲" if raw > 1.05 else "")
            ax.text(bar.get_x() + bar.get_width() / 2, vd + 0.01,
                    lbl, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_ylim(0, 1.20)
        ax.set_xticks(x)
        ax.set_xticklabels([MS.get(m, m) for m in models], fontsize=8)
        ax.set_title(GL[g], fontsize=10, fontweight="bold")
        ax.set_ylabel("Average AR", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.02, 0.98, f"n={gd[models[0]]['n_seeds']} seeds",
                transform=ax.transAxes, fontsize=8, va="top", color="#64748b")

    path = out_dir / "fig1_ar_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Figure 2: Violation rates ─────────────────────────────────────────────────

def fig_violations(
    groups_agg: dict[str, dict[str, dict]],
    out_dir:    Path,
) -> Path:
    fig, axes, pg = _fig3cols(groups_agg)
    fig.suptitle("Constraint Violation Rates  (cap + conflict)",
                 fontsize=12, fontweight="bold")

    for ax, g in zip(axes, pg):
        gd     = groups_agg[g]
        models = _models_in_group(gd)
        x      = np.arange(len(models))
        w      = 0.38

        cap_vals  = [gd[m]["cap_viol"]  for m in models]
        conf_vals = [gd[m]["conf_viol"] for m in models]

        ax.bar(x - w / 2, cap_vals,  w, color="#e74c3c", alpha=0.82,
               label="Cap viol.", edgecolor="white", linewidth=0.4)
        ax.bar(x + w / 2, conf_vals, w, color="#f39c12", alpha=0.82,
               label="Conflict viol.", edgecolor="white", linewidth=0.4)

        for xi, v in zip(x - w / 2, cap_vals):
            if v > 0.02:
                ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7)
        for xi, v in zip(x + w / 2, conf_vals):
            if v > 0.02:
                ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7)

        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([MS.get(m, m) for m in models], fontsize=8)
        ax.set_title(GL[g], fontsize=10, fontweight="bold")
        ax.set_ylabel("Violation Rate", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    path = out_dir / "fig2_violations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Figure 3: Success + Placement ────────────────────────────────────────────

def fig_success(
    groups_agg: dict[str, dict[str, dict]],
    out_dir:    Path,
) -> Path:
    fig, axes, pg = _fig3cols(groups_agg)
    fig.suptitle("Success / Failure Rate  (full placement + zero violations)",
                 fontsize=12, fontweight="bold")

    for ax, g in zip(axes, pg):
        gd     = groups_agg[g]
        models = _models_in_group(gd)
        x      = np.arange(len(models))

        sr = [gd[m]["success"] for m in models]
        fr = [gd[m]["failure"] for m in models]

        ax.bar(x, sr, color="#2ecc71", alpha=0.85, label="Success")
        ax.bar(x, fr, bottom=sr, color="#e74c3c", alpha=0.85, label="Failure")

        for i, (s, f) in enumerate(zip(sr, fr)):
            if s > 0.06:
                ax.text(i, s / 2, f"{s:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
            if f > 0.06:
                ax.text(i, s + f / 2, f"{f:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")

        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([MS.get(m, m) for m in models], fontsize=8)
        ax.set_title(GL[g], fontsize=10, fontweight="bold")
        ax.set_ylabel("Rate", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    path = out_dir / "fig3_success.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── Figure 4: Safety-Performance tradeoff ────────────────────────────────────

def fig_tradeoff(
    groups_agg: dict[str, dict[str, dict]],
    ilp_data:   dict[str, float | None],
    out_dir:    Path,
) -> Path:
    fig, axes, pg = _fig3cols(groups_agg, figw_per_col=5.5)
    fig.suptitle("Safety–Performance Frontier  (AR/ILP vs violation rate)",
                 fontsize=12, fontweight="bold")

    FEASIBLE_THR = 0.10   # 10% violation rate threshold

    for ax, g in zip(axes, pg):
        gd     = groups_agg[g]
        models = _models_in_group(gd)
        ilp    = ilp_data.get(g)

        total_viols = {m: gd[m]["cap_viol"] + gd[m]["conf_viol"] for m in models}
        ar_frac     = {m: (gd[m]["ar"] / ilp) if ilp else gd[m]["ar"] for m in models}

        x_max = max(max(total_viols.values()) * 1.2, FEASIBLE_THR * 4)
        ax.axvspan(0, FEASIBLE_THR, color="#d4edda", alpha=0.5)
        ax.axvspan(FEASIBLE_THR, x_max, color="#f8d7da", alpha=0.4)
        ax.axvline(FEASIBLE_THR, color="green", linestyle=":", linewidth=1.2)

        if ilp:
            ax.scatter([0], [1.0], marker="*", s=240, color="gold",
                       edgecolors="#8b6914", linewidth=0.8, zorder=6,
                       label=f"ILP = {ilp:.3f}")

        for m in models:
            xv = total_viols[m]
            yv = ar_frac[m]
            color = MODEL_COLORS.get(m, _DEFAULT_COLOR)
            ax.scatter([xv], [yv], s=90, color=color,
                       edgecolors="black", linewidth=0.6, zorder=5)
            ax.annotate(MS.get(m, m).replace("\n", " "),
                        (xv, yv), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, fontweight="bold", color=color)

        ax.set_xlim(0, x_max)
        y_all = list(ar_frac.values()) + ([1.0] if ilp else [])
        ax.set_ylim(min(y_all) - 0.05, max(y_all) + 0.08)
        ax.set_xlabel("Total violation rate  (cap + conflict)", fontsize=9)
        ax.set_ylabel("AR / ILP" if ilp else "AR", fontsize=9)
        ax.set_title(GL[g], fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        if ilp:
            ax.legend(fontsize=8, loc="lower right")

    path = out_dir / "fig4_tradeoff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── main entry ────────────────────────────────────────────────────────────────

def plot_experiment_summary(
    groups_agg: dict[str, dict[str, dict]],
    ilp_data:   dict[str, float | None],
    out_dir:    Path,
) -> list[Path]:
    """Generate all 4 summary figures into out_dir. Returns list of created paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(fig_ar_comparison(groups_agg, ilp_data, out_dir))
    paths.append(fig_violations(groups_agg, out_dir))
    paths.append(fig_success(groups_agg, out_dir))
    paths.append(fig_tradeoff(groups_agg, ilp_data, out_dir))
    return paths
