"""
Shared utilities for all problem run_all.py pipelines.

Functions here are identical across problem3-6, dqn, ddqn and are extracted
to avoid copy-paste duplication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pulp
import torch
import yaml


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
    from problem2_ilp.objects import ECU, SVC

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

    Design: N < M, multiple services must share ECUs (no uniqueness constraint).
    conflict_sets: list of lists of service indices — at most one from each subset per ECU.
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
    """Return (mean_ar, per_scenario_results) for all scenarios.
    Priority: 1) shared p2 cache  2) own local cache  3) compute from scratch.
    """
    from problem2_ilp.objects import ECU, SVC

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

    # ─ 1. Shared cache written by problem2_ilp/optimal_solution/main.py ─
    shared_cache = yaml_config.parent.parent / "results" / "ilp_cache.json"
    if shared_cache.exists():
        cache = _load_cache(shared_cache)
        if cache.get("key") == cache_key and len(cache.get("results", [])) == len(scenarios):
            print("    [cache] Loaded ILP results from shared p2 cache")
            results = cache["results"]
            ars = [r["avg_utilization"] for r in results if r.get("status") == "Optimal"]
            mean_ar = float(np.mean(ars)) if ars else 0.0
            print(f"    [cache] Feasible scenarios: {len(ars)}/{len(results)} — mean AR={mean_ar:.4f}")
            return mean_ar, results

    # ─ 2. Own local incremental cache ─
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "ilp_cache.json"
    results: list = []
    if cache_path.exists():
        cache = _load_cache(cache_path)
        if cache.get("key") == cache_key:
            results = cache.get("results", [])
            if len(results) == len(scenarios):
                print(f"    [cache] Loaded ILP results from {cache_path}")
                ars = [r["avg_utilization"] for r in results if r.get("status") == "Optimal"]
                mean_ar = float(np.mean(ars)) if ars else 0.0
                print(f"    [cache] Feasible scenarios: {len(ars)}/{len(results)} — mean AR={mean_ar:.4f}")
                return mean_ar, results
            print(f"    [cache] Resuming from scenario {len(results) + 1}")

    # ─ 3. Compute remaining, save after each ─
    for idx, sc in enumerate(scenarios[len(results):], start=len(results)):
        caps, reqs, conflict_sets = sc[0], sc[1], sc[2] if len(sc) > 2 else []
        ecus_sc = [ECU(f"ECU{i}", c) for i, c in enumerate(caps)]
        svcs_sc = [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)]
        res = solve_ilp(ecus_sc, svcs_sc, conflict_sets)
        results.append(res)
        print(f"    Scenario {idx + 1}: AR={res['avg_utilization']:.4f}  ({res['status']})")
        _save_cache(cache_path, {"key": cache_key, "results": results})

    print(f"    [cache] Saved to {cache_path}")
    ars = [r["avg_utilization"] for r in results if r.get("status") == "Optimal"]
    mean_ar = float(np.mean(ars)) if ars else 0.0
    print(f"    Feasible scenarios: {len(ars)}/{len(results)} — mean AR={mean_ar:.4f}")
    return mean_ar, results


# ── Plotting helper ───────────────────────────────────────────────────────────

def moving_avg(arr, w):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    return np.convolve(arr, np.ones(w) / w, mode="valid"), w - 1
