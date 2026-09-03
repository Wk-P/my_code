"""
Regenerate config_ecu_lt_svc.yaml with ALL scenarios guaranteed ILP-feasible.

Previous generator (see git history at v0.0.1) intentionally injected 20%
"infeasible" scenarios (one conflict set larger than N_ECUS, making the
conflict constraint unsatisfiable by pigeonhole). That's now removed —
every scenario here uses conflict sets sized <= N_ECUS, verified by greedy
graph colouring, and then re-verified by actually solving the ILP.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # scenario dir, for `ilp` package

import yaml
import datetime
import random

from ilp.objects import ECU, SVC
from shared.ilp_utils import solve_ilp

N_ECUS = 10   # ECU count < SVC count (ecu_lt_svc)
N_SVCS = 15
K_SETS = 10   # number of conflict sets per scenario
SEED   = 42


def _is_conflict_feasible(conflict_sets: list, M: int, N: int) -> bool:
    """Greedy graph colouring: True iff conflict constraints are satisfiable with N ECUs."""
    adj = [set() for _ in range(M)]
    for cs in conflict_sets:
        valid = [i for i in cs if i < M]
        for i in valid:
            for j in valid:
                if i != j:
                    adj[i].add(j)
    colors = [-1] * M
    for node in range(M):
        used = {colors[nb] for nb in adj[node] if colors[nb] >= 0}
        for c in range(N):
            if c not in used:
                colors[node] = c
                break
        if colors[node] == -1:
            return False
    return True


def _generate_capacity_and_services():
    ecu_capacity    = random.sample(range(50, 200, 5), N_ECUS)
    svc_requirement = random.sample(range(10, 100, 5), N_SVCS)
    return ecu_capacity, svc_requirement


def generate_feasible(scenario_id: int) -> dict:
    """Generate a scenario guaranteed feasible: conflict sets all <= N_ECUS
    (greedy-colourable), then double-checked by actually solving the ILP."""
    while True:
        ecu_capacity, svc_requirement = _generate_capacity_and_services()
        conflict_sets = [
            sorted(random.sample(range(N_SVCS), random.randint(2, N_ECUS)))
            for _ in range(K_SETS)
        ]
        if not _is_conflict_feasible(conflict_sets, N_SVCS, N_ECUS):
            continue
        ecus = [ECU(f"ECU{i}", c) for i, c in enumerate(ecu_capacity)]
        svcs = [SVC(f"SVC{i}", r) for i, r in enumerate(svc_requirement)]
        res = solve_ilp(ecus, svcs, conflict_sets)
        if res["status"] == "Optimal":
            break
    return _build_config(scenario_id, ecu_capacity, svc_requirement, conflict_sets)


def _build_config(scenario_id, ecu_capacity, svc_requirement, conflict_sets) -> dict:
    ecu_list = [ECU(f"ECU{i}", cap) for i, cap in enumerate(ecu_capacity)]
    svc_list = [SVC(f"SVC{i}", req) for i, req in enumerate(svc_requirement)]
    print(f"  Scenario {scenario_id:>3d} [feasible]: {N_ECUS} ECUs, {N_SVCS} SVCs")
    return {
        "name":          f"Scenario {scenario_id}",
        "feasible":      True,
        "generated_by":  "generate_config.py",
        "generated_on":  datetime.datetime.now().isoformat(),
        "ECUs":          [ecu.__dict__() for ecu in ecu_list],
        "SVCs":          [svc.__dict__() for svc in svc_list],
        "conflict_sets": conflict_sets,
    }


def write_config(scenarios: list, filename: str) -> None:
    config = {
        "generated_by":  "generate_config.py",
        "generated_on":  datetime.datetime.now().isoformat(),
        "feasible_rate": 1.0,
        "scenarios":     scenarios,
    }
    path = Path(__file__).parent / filename
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    n_feasible = sum(1 for s in scenarios if s.get("feasible"))
    print(f"\nSaved: {path.resolve()}")
    print(f"Total: {len(scenarios)}  |  Feasible: {n_feasible}  |  Infeasible: {len(scenarios) - n_feasible}")


if __name__ == "__main__":
    random.seed(SEED)
    n_total = 2000
    print(f"Generating {n_total} scenarios, all feasible ...")
    scenarios = [generate_feasible(i + 1) for i in range(n_total)]
    random.shuffle(scenarios)
    write_config(scenarios, "config_ecu_lt_svc.yaml")
