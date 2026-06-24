"""
P6 Environment — PPO + Best-fit Repair Heuristic.

Constraints:
    - Capacity or conflict violation → best-fit repair heuristic relocates the
      service to the valid ECU with the highest utilisation density (req/cap).
    - A repaired step incurs a small penalty (-0.1) to signal suboptimal choice.
    - If no valid ECU exists for repair → episode terminates with demand penalty.
    - Terminal bonus: +AR * (1 - repair_rate), rewarding episodes with fewer repairs.
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
        [2]          sum of remaining ECU capacity (normalised, clipped >= 0)
        [3]          sum of remaining service demand (normalised)
        [4]          fraction of ECUs with sufficient capacity AND no conflict
        [5]          fraction of services remaining
        [6:6+N]      initial capacity fraction per ECU
        [6+N:6+2N]   remaining capacity fraction per ECU
        [6+2N:6+3N]  conflict flag per ECU (1 = placing current svc here conflicts)
        [6+3N:6+4N]  valid-action flags (1 = sufficient capacity AND no conflict)
        [6+4N:6+4N+M] remaining service demands (sorted descending)

    Reward:
        +ru               valid assignment (agent chose correctly)
        +ru - 0.1         valid assignment after repair (agent chose poorly)
        demand_penalty    episode terminated — no repair possible
        terminal_bonus    +AR * (1 - repair_rate) at episode end
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
        self.repairs:            int
        self.cap_violations:     int
        self.conflict_violations: int
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

    # ── best-fit repair ──────────────────────────────────────────────────────
    def _best_fit_repair(self, svc_idx: int) -> int | None:
        """Return ECU index with highest req/cap among capacity-valid, conflict-free ECUs."""
        svc = self.services[svc_idx]
        valid = [
            j for j in range(self.N)
            if self.remaining_vms[j] >= svc.requirement
            and not self._has_conflict(j, svc_idx)
        ]
        if not valid:
            return None
        return max(valid, key=lambda j: svc.requirement / (self.initial_vms[j] + 1e-8))

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
        self.repairs             = 0
        self.cap_violations      = 0
        self.conflict_violations = 0
        return self._obs(), {}

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
            valid_flag = np.array(
                [(self.remaining_vms[j] >= svc.requirement) and (not self._has_conflict(j, self._step))
                 for j in range(self.N)],
                dtype=np.float32,
            )
            remaining_service_demand_sum = np.float32(
                sum(self.services[t].requirement for t in range(self._step, self.M)) / total_cap
            )
            remaining_services_count   = np.float32((self.M - self._step) / max(self.M, 1))
            remaining_usable_ecu_count = np.float32(np.sum(valid_flag) / max(self.N, 1))

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

        was_repaired = False
        if cap_violated or conflict_violated:
            repaired = self._best_fit_repair(self._step)
            if repaired is None:
                unplaced_demand = sum(self.services[i].requirement for i in range(self._step, self.M))
                penalty = -float(unplaced_demand) / (np.sum(self.initial_vms) + 1e-8)
                return self._obs(), penalty, True, False, {
                    "ar":                self.ar,
                    "step":              self._step,
                    "services_placed":   self._step,
                    "was_repaired":      False,
                    "repairs":           self.repairs,
                    "cap_violations":    self.cap_violations,
                    "conflict_violations": self.conflict_violations,
                    "repair_rate":       self.repairs / max(self._step, 1),
                }
            action = repaired
            was_repaired = True
            self.repairs += 1
            if cap_violated:
                self.cap_violations += 1
            if conflict_violated:
                self.conflict_violations += 1

        repair_penalty = -0.1 if was_repaired else 0.0
        ru = svc.requirement / (self.initial_vms[action] + 1e-8)
        self.remaining_vms[action] -= svc.requirement
        self.ecu_placements[action].add(self._step)
        self._total_ru += ru
        _active = sum(1 for j in range(self.N) if self.ecu_placements[j])
        self.ar = self._total_ru / _active
        self._step += 1

        done = self._step >= self.M
        terminal_bonus = 0.0
        if done:
            repair_rate = self.repairs / max(self.M, 1)
            terminal_bonus = self.ar * max(0.0, 1.0 - repair_rate)

        return self._obs(), float(ru + repair_penalty + terminal_bonus), done, False, {
            "ar":                  self.ar,
            "step":                self._step,
            "services_placed":     self._step,
            "was_repaired":        was_repaired,
            "repairs":             self.repairs,
            "cap_violations":      self.cap_violations,
            "conflict_violations": self.conflict_violations,
            "repair_rate":         self.repairs / self._step,
        }

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs | AR={self.ar:.4f} "
                  f"| repairs={self.repairs}")
        else:
            print(f"  Done | AR={self.ar:.4f} | repairs={self.repairs}/{self.M}")


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

    print("\n── Random policy run (repair heuristic active) ──")
    done = False
    while not done:
        a = env.action_space.sample()
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR     : {info['ar']:.4f}")
    print(f"Repairs      : {info['repairs']}/{M}")
    print(f"Repair rate  : {info['repair_rate']:.2%}")
