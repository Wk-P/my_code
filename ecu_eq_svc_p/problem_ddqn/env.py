"""
DDQN Environment — RL WITHOUT action masking.

Identical to DQNEnv; the distinction is in the training algorithm
(DDQN uses dual Q-networks to reduce overestimation).

Constraint handling:
    - Capacity violation → hard (episode terminates immediately with penalty).
    - Conflict violation → soft (recorded in info, does not terminate episode).
    - Multiple services may share an ECU (no uniqueness constraint).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_ilp.objects import ECU, SVC


class DDQNEnv(gym.Env):
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
        [6+3N:6+4N]  valid-action flags (1 = sufficient capacity)
        [6+4N:6+4N+M] remaining service demands (sorted descending)

    Reward:
        -remaining_services  on capacity violation (hard fail)
        +ru                  for valid assignment (req/cap)
        conflict violations are recorded in info only
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
        self.remaining_vms:   np.ndarray
        self.ecu_placements:  list[set]
        self.conflict_sets:   list[set]
        self.ar:              float
        self._step:           int
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
            valid_flag = (self.remaining_vms >= svc.requirement).astype(np.float32)
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
        svc = self.services[self._step]

        if self.remaining_vms[action] < svc.requirement:
            remaining_services = self.M - self._step
            return self._obs(), -float(remaining_services), True, False, {
                "ar":                  self.ar,
                "services_placed":     self._step,
                "violated":            True,
                "conflict_violations": self.conflict_violations,
            }

        conflict_violated = self._has_conflict(action, self._step)
        if conflict_violated:
            self.conflict_violations += 1
            remaining_services = self.M - self._step
            return self._obs(), -float(remaining_services), True, False, {
                "ar":                  self.ar,
                "services_placed":     self._step,
                "violated":            True,
                "conflict_violations": self.conflict_violations,
            }

        ru = svc.requirement / (self.initial_vms[action] + 1e-8)
        self.remaining_vms[action] -= svc.requirement
        self.ecu_placements[action].add(self._step)
        self._total_ru += ru
        _active = sum(1 for j in range(self.N) if self.ecu_placements[j])
        self.ar = self._total_ru / _active
        self._step += 1

        done = self._step >= self.M
        return self._obs(), float(ru), done, False, {
            "ar":                  self.ar,
            "step":                self._step,
            "services_placed":     self._step,
            "violated":            False,
            "conflict_violations": self.conflict_violations,
        }

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs "
                  f"| AR={self.ar:.4f} | conflict_viol={self.conflict_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f} | conflict_viol={self.conflict_violations}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random as rng
    rng.seed(42)
    N, M = 7, 4
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(rng.sample(range(50, 200, 5), N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(rng.sample(range(10, 100, 5), M))]

    print("ECU capacities :", [e.capacity for e in ecus])
    print("Service demands:", [s.requirement for s in services])

    env = DDQNEnv(ecus, services)
    obs, _ = env.reset()
    print(f"\nObs shape : {obs.shape}  (expected {4 * N + 6 + M})")

    print("\n── Valid greedy policy run ──")
    done = False
    while not done:
        svc = env.services[env._step]
        valid = [j for j in range(N) if env.remaining_vms[j] >= svc.requirement]
        if not valid:
            print("  No valid ECU!")
            break
        a = valid[0]
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR             : {info['ar']:.4f}")
    print(f"Services placed      : {info['services_placed']}/{M}")
    print(f"Conflict violations  : {info['conflict_violations']}")
