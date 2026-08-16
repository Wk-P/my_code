"""summarize.py — reads data/{lt,eq,gt}/results.json (each copied verbatim
from results/add_states/<scenario>/ppo_mask/<exp_id>/results.json) and
prints/saves a single cross-scenario summary table.

Deliberately does NOT merge or average across scenarios (lt/eq/gt have
different N/M and are not comparable on the same footing -- see
version/v2.5.0.md / project_thesis_ilp_rl memory for why cross-scenario
averaging is methodologically meaningless here). Each scenario's row stands
alone; this script only saves everyone the effort of opening 3 separate
JSON files by hand.

Usage: python summarize.py  (run from this scripts/ dir, or pass --root)
"""
import argparse
import json
from pathlib import Path


def load(scenario_dir: Path) -> dict:
    with open(scenario_dir / "results.json") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).parent.parent / "data")
    args = ap.parse_args()

    rows = []
    for sc in ["lt", "eq", "gt"]:
        sc_dir = args.root / sc
        if not (sc_dir / "results.json").exists():
            print(f"[skip] {sc}: no results.json under {sc_dir}")
            continue
        d = load(sc_dir)
        algo_key = next(
            k for k in d.keys()
            if k not in ("scenario", "prototype_scenario", "scenario_count",
                         "train_count", "test_count", "N", "M", "ilp",
                         "training", "created_at", "bc", "exp_id", "feasibility")
        )
        algo = d[algo_key]
        ilp_ar = d["ilp"]["ar"]
        rows.append({
            "scenario":     sc,
            "N":            d["N"],
            "M":            d["M"],
            "ilp_ar":       ilp_ar,
            "rl_ar_mean":   algo["ar_mean"],
            "rl_ar_std":    algo["ar_std"],
            "ar_gap":       round(ilp_ar - algo["ar_mean"], 6),
            "success_rate": algo["success_rate"],
            "attempts_mean": algo.get("attempts_mean"),
            "total_steps":  d["training"]["total_steps"],
        })

    if not rows:
        print("No data found -- did you copy results.json into data/{lt,eq,gt}/ first?")
        return

    hdr = f"{'scenario':<10}{'N':>4}{'M':>4}{'ilp_ar':>10}{'rl_ar':>10}{'ar_gap':>10}{'success%':>10}{'avg_try':>9}{'steps':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['scenario']:<10}{r['N']:>4}{r['M']:>4}{r['ilp_ar']:>10.4f}"
            f"{r['rl_ar_mean']:>10.4f}{r['ar_gap']:>10.4f}"
            f"{r['success_rate']*100:>9.1f}%{r['attempts_mean'] or float('nan'):>9.2f}"
            f"{r['total_steps']:>12,}"
        )

    out_path = args.root.parent / "summary.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
