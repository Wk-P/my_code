"""
P6 Environment — RL WITH strict constraint enforcement (both hard).

Design intent:
    • Capacity violation → hard (episode terminates immediately with penalty).
    • Conflict violation → hard (episode terminates immediately with penalty).
    • Multiple services may share an ECU; uniqueness is NOT a constraint.
    • Reward equals the exact utilisation contribution of each valid step.
    • Scenarios: reset() randomly picks one of the provided scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_ilp.objects import ECU, SVC


class P6Env(gym.Env):
    """
    Observation (shape: 4N+6+M):
        [0]          current service demand (normalised)
        [1]          current cumulative AR
        [2]          sum of remaining ECU capacity (normalised, clipped ≥ 0)
        [3]          sum of remaining service demand (normalised)
        [4]          fraction of ECUs with sufficient capacity for current service
        [5]          fraction of services remaining
        [6:6+N]      initial capacity fraction per ECU
        [6+N:6+2N]   remaining capacity fraction per ECU
        [6+2N:6+3N]  conflict flag per ECU
        [6+3N:6+4N]  valid-action flags (1 = sufficient capacity AND no conflict)
        [6+4N:6+4N+M] remaining service demands (sorted descending)

    Reward:
        +ru   for each valid assignment (exact utilisation contribution)
        demand_penalty when episode terminates early (capacity or conflict violation)
    """

    metadata = {"render_modes": []}

    def __init__(self, ecus: list[ECU], services: list[SVC], scenarios=None):
        super().__init__()
        self._scenarios = scenarios
        self.ecus     = ecus
        self.services = services
        self.N = len(ecus)
        self.M = len(services)

        self.action_space = gym.spaces.Discrete(self.N)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4 * self.N + 6 + self.M,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms:  np.ndarray
        self.ecu_placements: list[set]
        self.conflict_sets:  list[set]
        self.ar:             float
        self._step:          int
        self.reset()

    # ── conflict helpers ─────────────────────────────────────────────────────
    def _init_conflict_sets(self, K: int = 10) -> list[set]:
        sets = []
        for _ in range(K):
            j = random.randint(2, self.M)
            sets.append(set(random.sample(range(self.M), j)))
        return sets

    def _has_conflict(self, ecu_idx: int, svc_idx: int) -> bool:
        placed = self.ecu_placements[ecu_idx]
        if not placed:
            return False
        for subset in self.conflict_sets:
            if svc_idx in subset:
                for p in placed:
                    if p in subset:
                        return True
        return False

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs, _cs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
            self.conflict_sets = [set(s) for s in _cs]
        else:
            self.conflict_sets = self._init_conflict_sets()
        self.services = sorted(self.services, key=lambda s: s.requirement, reverse=True)
        self.remaining_vms   = self.initial_vms.copy()
        self.ecu_placements  = [set() for _ in range(self.N)]
        self.ar              = 0.0
        self._total_ru       = 0.0
        self._step           = 0
        self.capacity_violations = 0
        self.conflict_violations = 0
        return self._obs(), {}

    # ── action mask ──────────────────────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        if self._step >= self.M:
            return np.zeros(self.N, dtype=bool)
        svc = self.services[self._step]
        return np.array(
            [(self.remaining_vms[j] >= svc.requirement) and (not self._has_conflict(j, self._step))
             for j in range(self.N)],
            dtype=bool,
        )

    # ── observation ──────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        max_cap   = float(np.max(self.initial_vms) + 1e-8)
        total_cap = float(np.sum(self.initial_vms) + 1e-8)

        if self._step >= self.M:
            service_demand_norm          = np.float32(0.0)
            conflict_flag                = np.zeros(self.N, dtype=np.float32)
            valid_flag                   = np.zeros(self.N, dtype=np.float32)
            remaining_service_demand_sum = np.float32(0.0)
            remaining_services_count     = np.float32(0.0)
            remaining_usable_ecu_count   = np.float32(0.0)
        else:
            svc = self.services[self._step]
            service_demand_norm = np.float32(svc.requirement / max_cap)
            conflict_flag = np.array(
                [float(self._has_conflict(j, self._step)) for j in range(self.N)],
                dtype=np.float32,
            )
            valid_flag = self.action_masks().astype(np.float32)
            remaining_service_demand_sum = np.float32(
                sum(self.services[t].requirement for t in range(self._step, self.M)) / total_cap
            )
            remaining_services_count   = np.float32((self.M - self._step) / max(self.M, 1))
            remaining_usable_ecu_count = np.float32(
                np.sum(self.remaining_vms >= svc.requirement) / max(self.N, 1)
            )

        remaining_abs_norm = self.remaining_vms / max_cap
        initial_cap_pct    = self.initial_vms / max_cap
        remaining_usable_capacity_sum = np.float32(
            np.sum(np.clip(self.remaining_vms, 0.0, None)) / total_cap
        )
        remaining_svcs = np.array(
            [self.services[t].requirement / max_cap if t >= self._step else 0.0
             for t in range(self.M)],
            dtype=np.float32,
        )

        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            np.array([remaining_usable_capacity_sum], dtype=np.float32),
            np.array([remaining_service_demand_sum], dtype=np.float32),
            np.array([remaining_usable_ecu_count], dtype=np.float32),
            np.array([remaining_services_count], dtype=np.float32),
            initial_cap_pct,
            remaining_abs_norm,
            conflict_flag,
            valid_flag,
            remaining_svcs,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert 0 <= action < self.N, f"Invalid action {action}"
        svc = self.services[self._step]

        cap_violated      = bool(self.remaining_vms[action] < svc.requirement)
        conflict_violated = self._has_conflict(action, self._step)

        if cap_violated:
            self.capacity_violations += 1
        if conflict_violated:
            self.conflict_violations += 1

        if cap_violated or conflict_violated:
            remaining_services = self.M - self._step
            unplaced_demand = sum(self.services[i].requirement for i in range(self._step, self.M))
            penalty = -float(unplaced_demand) / (np.sum(self.initial_vms) + 1e-8)
            return self._obs(), penalty, True, False, {
                "ar":                  self.ar,
                "step":                self._step,
                "services_placed":     self._step,
                "capacity_violations": self.capacity_violations,
                "conflict_violations": self.conflict_violations,
                "total_violations":    self.capacity_violations + self.conflict_violations,
                "violation_rate":      1.0 if remaining_services > 0 else 0.0,
            }

        ru = svc.requirement / (self.initial_vms[action] + 1e-8)
        self.remaining_vms[action] -= svc.requirement
        self.ecu_placements[action].add(self._step)
        self._total_ru += ru
        _active = sum(1 for j in range(self.N) if self.ecu_placements[j])
        self.ar = self._total_ru / _active
        self._step += 1

        done  = self._step >= self.M
        total_viol = self.capacity_violations + self.conflict_violations
        info = {
            "ar":                  self.ar,
            "step":                self._step,
            "services_placed":     self._step,
            "capacity_violations": self.capacity_violations,
            "conflict_violations": self.conflict_violations,
            "total_violations":    total_viol,
            "violation_rate":      total_viol / self._step,
        }
        return self._obs(), float(ru), done, False, info

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs | AR={self.ar:.4f} "
                  f"| cap_viol={self.capacity_violations} conflict_viol={self.conflict_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f} "
                  f"| cap_viol={self.capacity_violations} conflict_viol={self.conflict_violations}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    N, M = 7, 4
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(random.sample(range(50, 200, 5), N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(random.sample(range(10, 100, 5), M))]

    print("ECU capacities :", [e.capacity for e in ecus])
    print("Service demands:", [s.requirement for s in services])

    env = P6Env(ecus, services)
    obs, _ = env.reset()
    print(f"\nObs shape : {obs.shape}  (expected {4*N + 6 + M})")

    print("\n── Valid greedy policy run ──")
    done = False
    while not done:
        mask  = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            print("  No valid actions remaining!")
            break
        a = int(valid[0])
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR             : {info['ar']:.4f}")
    print(f"Capacity violations  : {info['capacity_violations']}")
    print(f"Conflict violations  : {info['conflict_violations']}")
