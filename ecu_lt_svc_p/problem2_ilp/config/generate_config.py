import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
from problem2_ilp.objects import ECU, SVC
import datetime
import random

N_ECUS = 10  # ECU count < SVC count (ecu_lt_svc)
N_SVCS = 15

FEASIBLE_RATE = 0.8   # 80% feasible, 20% infeasible
K_SETS        = 10    # number of conflict sets per scenario


def _is_conflict_feasible(conflict_sets: list[list[int]], M: int, N: int) -> bool:
    """Greedy graph colouring: True iff conflict constraints are satisfiable with N ECUs."""
    adj: list[set[int]] = [set() for _ in range(M)]
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
    """Generate a scenario guaranteed feasible for conflict constraints.

    All conflict sets have size ≤ N_ECUS, then verified with greedy colouring.
    Retries until a feasible conflict structure is found.
    """
    ecu_capacity, svc_requirement = _generate_capacity_and_services()
    while True:
        conflict_sets = [
            random.sample(range(N_SVCS), random.randint(2, N_ECUS))
            for _ in range(K_SETS)
        ]
        if _is_conflict_feasible(conflict_sets, N_SVCS, N_ECUS):
            break
    return _build_config(scenario_id, ecu_capacity, svc_requirement, conflict_sets, feasible=True)


def generate_infeasible(scenario_id: int) -> dict:
    """Generate a scenario guaranteed infeasible for conflict constraints.

    Forces at least one conflict set with size > N_ECUS; that set alone
    requires more ECUs than available, making the scenario unsolvable.
    """
    ecu_capacity, svc_requirement = _generate_capacity_and_services()
    conflict_sets = []
    # One set with size in (N_ECUS, N_SVCS] — guaranteed infeasible
    conflict_sets.append(random.sample(range(N_SVCS), random.randint(N_ECUS + 1, N_SVCS)))
    # Remaining sets unrestricted
    for _ in range(K_SETS - 1):
        conflict_sets.append(random.sample(range(N_SVCS), random.randint(2, N_SVCS)))
    return _build_config(scenario_id, ecu_capacity, svc_requirement, conflict_sets, feasible=False)


def _build_config(scenario_id, ecu_capacity, svc_requirement, conflict_sets, feasible: bool) -> dict:
    ecu_list = [ECU(f"ECU{i}", cap) for i, cap in enumerate(ecu_capacity)]
    svc_list = [SVC(f"SVC{i}", req) for i, req in enumerate(svc_requirement)]
    label = "feasible" if feasible else "infeasible"
    print(f"  Scenario {scenario_id:>3d} [{label}]: {N_ECUS} ECUs, {N_SVCS} SVCs")
    return {
        "name":          f"Scenario {scenario_id}",
        "feasible":      feasible,
        "generated_by":  "generate_config.py",
        "generated_on":  datetime.datetime.now().isoformat(),
        "ECUs":          [ecu.__dict__() for ecu in ecu_list],
        "SVCs":          [svc.__dict__() for svc in svc_list],
        "conflict_sets": conflict_sets,
    }


def write_config(scenarios: list[dict], filename: str | None = None) -> None:
    config = {
        "generated_by":    "generate_config.py",
        "generated_on":    datetime.datetime.now().isoformat(),
        "feasible_rate":   FEASIBLE_RATE,
        "scenarios":       scenarios,
    }
    path_config_directory = Path(__file__).parent
    if filename is not None:
        new_file_path = path_config_directory / filename
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file_path = path_config_directory / f"config_{ts}.yaml"

    with open(new_file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    n_feasible   = sum(1 for s in scenarios if s.get("feasible"))
    n_infeasible = len(scenarios) - n_feasible
    print(f"\nSaved: {new_file_path.resolve()}")
    print(f"Total  : {len(scenarios)}  |  Feasible: {n_feasible} ({n_feasible/len(scenarios)*100:.0f}%)  |  Infeasible: {n_infeasible}")


if __name__ == "__main__":
    n_total      = 200
    n_feasible   = int(n_total * FEASIBLE_RATE)   # 160
    n_infeasible = n_total - n_feasible            # 40

    print(f"Generating {n_total} scenarios  ({n_feasible} feasible + {n_infeasible} infeasible) ...")
    scenarios: list[dict] = []

    print("\n[Feasible]")
    for i in range(n_feasible):
        scenarios.append(generate_feasible(i + 1))

    print("\n[Infeasible]")
    for i in range(n_infeasible):
        scenarios.append(generate_infeasible(n_feasible + i + 1))

    random.shuffle(scenarios)
    write_config(scenarios, "config_ecu_lt_svc.yaml")
