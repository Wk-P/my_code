"""
Shared utilities for all scenario run_all.py pipelines (ppo, ppo_mask,
ppo_lagrangian, ppo_opt, dqn, ddqn — across eq/gt/lt scenarios).

Previously this lived as three near-identical copies under
scenarios/{eq,gt,lt}/run_utils.py; consolidated here to avoid drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pulp
import torch
import yaml

from shared.paths import content_hash, results_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(description: str = "Run training pipeline") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override C.TOTAL_STEPS for quick smoke tests.",
    )
    return parser.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def resolve_device(cfg: str) -> str:
    if cfg != "auto":
        return cfg
    if torch.cuda.is_available():
        print(f"[CUDA] {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[CPU] No CUDA GPU, using CPU")
    return "cpu"


# ── Scenario loader ───────────────────────────────────────────────────────────

def load_scenario(yaml_config: Path, scenario_idx: int, scenarios: list):
    """Load the prototype scenario and return (ecus, services, scope_str, name)."""
    from ilp.objects import ECU, SVC

    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    scenario = data["scenarios"][scenario_idx]
    ecus     = [ECU(s["name"], s["capacity"])    for s in scenario["ECUs"]]
    services = [SVC(s["name"], s["requirement"]) for s in scenario["SVCs"]]
    N, M = len(ecus), len(services)
    scenario_scope = f"All {len(scenarios)} Scenarios"
    print(f"Loaded scenario pool: {scenario_scope}  |  N={N} ECUs  M={M} SVCs")
    print(f"  Prototype scenario: {scenario['name']} (idx={scenario_idx})")
    print(f"  Prototype ECU capacities : {[e.capacity for e in ecus]}")
    print(f"  Prototype SVC requirements: {[s.requirement for s in services]}")
    return ecus, services, scenario_scope, scenario["name"]


# ── ILP solver ────────────────────────────────────────────────────────────────

def solve_ilp(ecus, services, conflict_sets=None) -> dict:
    """Solve the assignment ILP via PuLP; return avg_utilization and allocation.

    N and M may be in any relation; multiple services may share one ECU
    (no uniqueness constraint). conflict_sets: list of lists of service
    indices — at most one from each subset per ECU.
    AR = total_utilization / active_ecus  (average ECU utilization).
    """
    N, M = len(ecus), len(services)
    e_list = [e.capacity    for e in ecus]
    n_list = [s.requirement for s in services]

    prob = pulp.LpProblem("ILP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(M), range(N)), cat="Binary")

    # Objective: maximise sum of req/cap (= M * AR)
    prob += pulp.lpSum(x[i][j] * n_list[i] / e_list[j]
                       for i in range(M) for j in range(N))
    # Each service assigned to exactly one ECU
    for i in range(M):
        prob += pulp.lpSum(x[i][j] for j in range(N)) == 1
    # Capacity constraint per ECU (multiple services allowed)
    for j in range(N):
        prob += pulp.lpSum(x[i][j] * n_list[i] for i in range(M)) <= e_list[j]
    # Hard-infeasible assignments
    for i in range(M):
        for j in range(N):
            if n_list[i] > e_list[j]:
                prob += x[i][j] == 0
    # Conflict constraints: at most one service from each conflict subset per ECU
    if conflict_sets:
        for k, subset in enumerate(conflict_sets):
            valid = [i for i in subset if i < M]
            if len(valid) >= 2:
                for j in range(N):
                    prob += pulp.lpSum(x[i][j] for i in valid) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    alloc = {}
    for j in range(N):
        svcs = [i for i in range(M) if pulp.value(x[i][j]) is not None and pulp.value(x[i][j]) > 0.5]
        if svcs:
            alloc[j] = {
                "services":    svcs,
                "utilization": sum(n_list[i] for i in svcs) / e_list[j],
                "capacity":    e_list[j],
                "demand":      sum(n_list[i] for i in svcs),
            }
    total_util = pulp.value(prob.objective) or 0.0
    # AR = total_util / active_ecus  (average ECU utilization)
    avg_util   = total_util / len(alloc) if alloc else 0.0
    return {
        "status":            pulp.LpStatus[prob.status],
        "avg_utilization":   avg_util,
        "total_utilization": total_util,
        "active_ecus":       len(alloc),
        "allocation":        alloc,
    }


def solve_ilp_all_scenarios(yaml_config: Path, scenarios: list, outdir: Path):
    """Return (mean_ar, per_scenario_results) for all scenarios, cached.

    `outdir` is expected to be results/<scenario>/<algo>/ (see shared/paths.py);
    the scenario name is derived from it. The ILP result only depends on the
    scenario config, not on which algo is asking, so every algo shares one
    content-addressed cache file at results/<scenario>/ilp/<hash>.json
    (hash = content_hash(cache_key)) instead of each algo keeping its own copy.
    """
    from ilp.objects import ECU, SVC

    cache_key = f"{yaml_config.name}__n{len(scenarios)}"

    def _load_cache(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cache(path, data):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.replace(path)

    def _feasible_mean(results):
        ars = [r["avg_utilization"] for r in results if r.get("status") == "Optimal"]
        mean_ar = float(np.mean(ars)) if ars else 0.0
        return mean_ar, len(ars)

    scenario_name = outdir.parent.name
    ilp_dir = results_dir(scenario_name, "ilp")
    ilp_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ilp_dir / f"{content_hash(cache_key)}.json"

    results: list = []
    if cache_path.exists():
        cache = _load_cache(cache_path)
        if cache.get("key") == cache_key:
            results = cache.get("results", [])
            if len(results) == len(scenarios):
                mean_ar, n_feasible = _feasible_mean(results)
                print(f"    [cache] Loaded ILP results from {cache_path}")
                print(f"    [cache] Feasible scenarios: {n_feasible}/{len(results)} — mean AR={mean_ar:.4f}")
                return mean_ar, results
            print(f"    [cache] Resuming from scenario {len(results) + 1}")

    # Compute remaining, save after each
    for idx, sc in enumerate(scenarios[len(results):], start=len(results)):
        caps, reqs, conflict_sets = sc[0], sc[1], sc[2] if len(sc) > 2 else []
        ecus_sc = [ECU(f"ECU{i}", c) for i, c in enumerate(caps)]
        svcs_sc = [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)]
        res = solve_ilp(ecus_sc, svcs_sc, conflict_sets)
        results.append(res)
        print(f"    Scenario {idx + 1}: AR={res['avg_utilization']:.4f}  ({res['status']})")
        _save_cache(cache_path, {"key": cache_key, "results": results})

    print(f"    [cache] Saved to {cache_path}")
    mean_ar, n_feasible = _feasible_mean(results)
    print(f"    Feasible scenarios: {n_feasible}/{len(results)} — mean AR={mean_ar:.4f}")
    return mean_ar, results


# ── Feasibility checker ───────────────────────────────────────────────────────

def check_scenario_feasibility(scenarios: list) -> dict:
    """Check each scenario for ILP feasibility. Returns counts and indices."""
    from ilp.objects import ECU, SVC
    feasible, infeasible = [], []
    for idx, sc in enumerate(scenarios):
        caps, reqs, conflict_sets = sc[0], sc[1], sc[2] if len(sc) > 2 else []
        ecus_sc = [ECU(f"ECU{i}", c) for i, c in enumerate(caps)]
        svcs_sc = [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)]
        res = solve_ilp(ecus_sc, svcs_sc, conflict_sets)
        if res["status"] == "Optimal" and res["active_ecus"] > 0:
            feasible.append(idx)
        else:
            infeasible.append(idx)
    return {
        "total": len(scenarios),
        "feasible": len(feasible),
        "infeasible": len(infeasible),
        "infeasible_indices": infeasible,
    }


# ── Plotting helper ───────────────────────────────────────────────────────────

def moving_avg(arr, w):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    return np.convolve(arr, np.ones(w) / w, mode="valid"), w - 1
