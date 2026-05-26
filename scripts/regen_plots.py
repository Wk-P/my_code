"""
regen_plots.py — Fix per-seed test_results.png and generate cross-seed summary figures.

Usage:
    python scripts/regen_plots.py                  # process all tagged experiments
    python scripts/regen_plots.py --run-id <id>    # process one specific experiment
    python scripts/regen_plots.py --reports-dir reports/ecu_exp_20260526_xxx  # new-style dir
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GROUPS       = ["eq", "lt", "gt"]
LEGACY_ROOTS = [ROOT / "results_v2", ROOT / "results"]
REPORTS_ROOT = ROOT / "reports"
FIGURES_ROOT = ROOT / "figures"


# ── ILP loader ────────────────────────────────────────────────────────────────

def _load_ilp(group: str) -> float | None:
    import json, numpy as np
    GROUP_ENV = {"lt": "ecu_lt_svc_p", "eq": "ecu_eq_svc_p", "gt": "ecu_gt_svc_p"}
    ILP_PROBE = ["problem4_ppo_mask", "problem3_ppo", "problem_dqn", "problem6_ppo_opt"]
    env_dir = ROOT / GROUP_ENV.get(group, "")
    for prob in ILP_PROBE:
        cache = env_dir / prob / "results" / "ilp_cache.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                ars = [r["avg_utilization"] for r in data.get("results", [])
                       if "avg_utilization" in r]
                if ars:
                    return round(float(np.mean(ars)), 4)
            except Exception:
                pass
    return None


# ── Regenerate per-seed test_results.png ─────────────────────────────────────

def _regen_one_seed(seed_dir: Path, group: str, seed_num: int | None = None):
    """Re-plot test_results.png from existing test_results_agg.csv."""
    # Find agg CSV (new or legacy path)
    for candidate in [
        seed_dir / "test_results" / "data" / "test_results_agg.csv",
        seed_dir / "test_results_agg.csv",
    ]:
        if candidate.exists():
            agg_csv = candidate
            break
    else:
        print(f"  [skip] no agg csv in {seed_dir}")
        return

    # Determine output path
    for fig_candidate in [
        seed_dir / "test_results" / "figures",
        seed_dir,
    ]:
        if (seed_dir / "test_results" / "data").exists():
            fig_out_dir = seed_dir / "test_results" / "figures"
        else:
            fig_out_dir = seed_dir
        break
    fig_out_dir.mkdir(parents=True, exist_ok=True)

    # Read agg CSV → dict expected by plot_test_results
    agg: dict[str, dict] = {}
    try:
        with open(agg_csv, newline="") as f:
            for row in csv.DictReader(f):
                agg[row["model"]] = {k: float(v) for k, v in row.items() if k != "model"}
    except Exception as e:
        print(f"  [error] reading {agg_csv}: {e}")
        return

    # Set up sys.path for the group so the correct plots module is loaded
    import importlib
    group_pkg = ROOT / f"ecu_{group}_svc_p"
    if str(group_pkg) not in sys.path:
        sys.path.insert(0, str(group_pkg))

    # Evict cached module so we re-import from the right group package
    for mod in list(sys.modules):
        if "experiment.plots" in mod or mod == "experiment":
            del sys.modules[mod]

    from experiment.plots import plot_test_results
    print(f"  Regenerating {seed_dir.name} → {fig_out_dir.name}/test_results.png")
    plot_test_results(agg, fig_out_dir, seed=seed_num)


# ── Collect seed dirs for a run ───────────────────────────────────────────────

def _seed_dirs_legacy(run_id: str) -> dict[str, list[Path]]:
    """Return {group: [seed_dirs]} for a legacy run_id."""
    result: dict[str, list[Path]] = {}
    for group in GROUPS:
        dirs = []
        for legacy_root in LEGACY_ROOTS:
            gdir = legacy_root / group
            if not gdir.exists():
                continue
            for sd in sorted(gdir.glob("seed_*")):
                rid_f = sd / ".run_id"
                if rid_f.exists() and rid_f.read_text().strip() == run_id:
                    dirs.append(sd)
        if dirs:
            result[group] = dirs
    return result


def _seed_dirs_reports(exp_dir: Path) -> dict[str, list[Path]]:
    """Return {group: [seed_dirs]} for a reports/ experiment directory."""
    result: dict[str, list[Path]] = {}
    for group in GROUPS:
        seeds_dir = exp_dir / group.upper() / "seeds"
        if not seeds_dir.exists():
            continue
        dirs = sorted([d for d in seeds_dir.iterdir() if d.is_dir()],
                      key=lambda p: int(p.name) if p.name.isdigit() else 0)
        if dirs:
            result[group] = dirs
    return result


def _summary_out_dir(run_id: str, exp_dir: Path | None) -> Path:
    """Where to save the 4 summary figures."""
    if exp_dir is not None:
        return exp_dir / "results" / "figures"
    return FIGURES_ROOT / run_id


# ── Process one experiment ────────────────────────────────────────────────────

def process_experiment(run_id: str, seed_dirs: dict[str, list[Path]],
                       exp_dir: Path | None = None):
    print(f"\n{'='*60}")
    print(f"  Experiment: {run_id}")
    print(f"  Groups: {list(seed_dirs.keys())}")
    print(f"{'='*60}")

    # 1. Regenerate per-seed test_results.png
    for group, dirs in seed_dirs.items():
        print(f"\n  [group={group}] Regenerating per-seed figures …")
        for sd in dirs:
            seed_num = None
            try:
                seed_num = int(sd.name.split("_")[-1])
            except ValueError:
                try:
                    seed_num = int(sd.name)
                except ValueError:
                    pass
            _regen_one_seed(sd, group, seed_num)

    # 2. Generate per-group cross-seed summary figures
    print(f"\n  Generating cross-seed summary figures …")
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))
    for mod in list(sys.modules):
        if mod.startswith("experiment"):
            del sys.modules[mod]
    from experiment.plot_summary import load_groups_agg, plot_experiment_summary

    for group, dirs in seed_dirs.items():
        group_agg = load_groups_agg({group: dirs})
        if not group_agg:
            continue
        ilp_val = None
        if exp_dir is not None:
            import json as _json
            ilp_json = exp_dir / group.upper() / "results" / "ilp.json"
            if ilp_json.exists():
                try:
                    ilp_val = float(_json.loads(ilp_json.read_text())["ilp_ar"])
                except Exception:
                    pass
        if ilp_val is None:
            ilp_val = _load_ilp(group)
        ilp_data = {group: ilp_val}
        if exp_dir is not None:
            out_dir = exp_dir / group.upper() / "results" / "figures"
        else:
            out_dir = FIGURES_ROOT / run_id / group.upper()
        print(f"  [{group.upper()}] → {out_dir}")
        plot_experiment_summary(group_agg, ilp_data, out_dir)

    print(f"\n  Done: {run_id}")


# ── Discover all tagged legacy experiments ────────────────────────────────────

def discover_legacy_runs() -> dict[str, dict[str, list[Path]]]:
    runs: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for group in GROUPS:
        for legacy_root in LEGACY_ROOTS:
            gdir = legacy_root / group
            if not gdir.exists():
                continue
            for sd in sorted(gdir.glob("seed_*")):
                rid_f = sd / ".run_id"
                if rid_f.exists() and (sd / "test_results_agg.csv").exists():
                    rid = rid_f.read_text().strip()
                    runs[rid][group].append(sd)
    return {rid: dict(gd) for rid, gd in runs.items()}


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",      type=str, default=None,
                        help="Process one specific legacy run_id.")
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help="Process one reports/ experiment directory.")
    parser.add_argument("--all",         action="store_true",
                        help="Process all discovered legacy experiments (default).")
    args = parser.parse_args()

    if args.reports_dir:
        exp_dir  = args.reports_dir.resolve()
        rid_f    = exp_dir / "run_info.json"
        if rid_f.exists():
            import json
            run_id = json.loads(rid_f.read_text()).get("run_id", exp_dir.name)
        else:
            run_id = exp_dir.name
        seed_dirs = _seed_dirs_reports(exp_dir)
        process_experiment(run_id, seed_dirs, exp_dir)

    elif args.run_id:
        seed_dirs = _seed_dirs_legacy(args.run_id)
        if not seed_dirs:
            print(f"No data found for run_id={args.run_id}")
            sys.exit(1)
        process_experiment(args.run_id, seed_dirs, exp_dir=None)

    else:
        # Default: all legacy experiments
        all_runs = discover_legacy_runs()
        if not all_runs:
            print("No tagged experiments found in results/ or results_v2/.")
            sys.exit(0)
        print(f"Found {len(all_runs)} experiment(s): {list(all_runs.keys())}")
        for run_id, seed_dirs in all_runs.items():
            process_experiment(run_id, seed_dirs, exp_dir=None)

    print("\nAll done.")


if __name__ == "__main__":
    main()
