"""
DDQN Environment — heavy penalty for both capacity and conflict violations.

N < M: each ECU hosts multiple services.

Identical constraint-enforcement strategy to DQNEnv.  The algorithm
distinction (dual Q-networks to reduce overestimation) is in the trainer,
not the environment.

Constraints:
    - Capacity violation → heavy penalty (-2.0), placement still proceeds.
    - Conflict violation → heavy penalty (-2.0), placement still proceeds.
    - A well-trained agent should achieve zero violations on feasible scenarios.
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
    Each episode assigns M services to N ECUs (N < M), one service per step.

    Observation (shape: 5N+6+M):
        [0]          current service demand (normalised)
        [1]          current cumulative AR
        [2]          sum of remaining ECU capacity (normalised, clipped ≥ 0)
        [3]          sum of remaining service demand (normalised)
        [4]          fraction of ECUs with sufficient capacity for current service
        [5]          fraction of services remaining
        [6:6+N]      initial capacity fraction per ECU
        [6+N:6+2N]   remaining capacity fraction per ECU (may be negative)
        [6+2N:6+3N]  conflict flag per ECU (1 = placing current svc here violates a conflict set)
        [6+3N:6+4N]  ECU allowed fraction (fraction of SVCs still placeable without conflict)
        [6+4N:6+5N]  valid-action flags (1 = sufficient capacity)
        [6+5N:6+5N+M] remaining service demands (sorted descending)

    Reward:
        +ru        valid assignment (req/cap_i)
        -2.0       per capacity violation
        -2.0       per conflict violation
        (penalties are additive; both violated → -4.0)
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
            low=-1.0, high=1.0, shape=(5 * self.N + 6 + self.M,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms:   np.ndarray
        self.ecu_placements:  list[set]
        self.conflict_sets:   list[set]
        self.ecu_allowed:     list[set]
        self.ar:              float
        self._step:           int
        self.capacity_violations: int
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
        return svc_idx not in self.ecu_allowed[ecu_idx]

    def _update_ecu_allowed(self, ecu_idx: int, svc_idx: int) -> None:
        for subset in self.conflict_sets:
            if svc_idx in subset:
                self.ecu_allowed[ecu_idx] -= (subset - {svc_idx})

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs, _cs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
            sorted_order = sorted(range(len(self.services)),
                                  key=lambda i: self.services[i].requirement, reverse=True)
            orig_to_new = {orig: new for new, orig in enumerate(sorted_order)}
            self.conflict_sets = [
                {orig_to_new[i] for i in cs if i < len(self.services)} for cs in _cs
            ]
        else:
            self.conflict_sets = self._init_conflict_sets()
        self.services = sorted(self.services, key=lambda s: s.requirement, reverse=True)
        self.remaining_vms   = self.initial_vms.copy()
        self.ecu_placements  = [set() for _ in range(self.N)]
        self.ecu_allowed     = [set(range(self.M)) for _ in range(self.N)]
        self.ar              = 0.0
        self._total_ru       = 0.0
        self._step           = 0
        self.capacity_violations = 0
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

        remaining_abs_norm = np.clip(self.remaining_vms, -max_cap, max_cap) / max_cap
        initial_cap_pct    = self.initial_vms / max_cap
        remaining_usable_capacity_sum = np.float32(
            np.sum(np.clip(self.remaining_vms, 0.0, None)) / total_cap
        )
        remaining_svcs = np.array(
            [self.services[t].requirement / max_cap if t >= self._step else 0.0
             for t in range(self.M)],
            dtype=np.float32,
        )

        ecu_allowed_frac = np.array(
            [len(self.ecu_allowed[j]) / self.M for j in range(self.N)],
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
            ecu_allowed_frac,
            valid_flag,
            remaining_svcs,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        svc = self.services[self._step]

        cap_violated      = bool(self.remaining_vms[action] < svc.requirement)
        conflict_violated = self._has_conflict(action, self._step)

        if cap_violated:
            self.capacity_violations += 1
        if conflict_violated:
            self.conflict_violations += 1

        cap_penalty      = -2.0 if cap_violated else 0.0
        conflict_penalty = -2.0 if conflict_violated else 0.0
        ru = 0.0 if (cap_violated or conflict_violated) else svc.requirement / (self.initial_vms[action] + 1e-8)

        # Always place; remaining_vms may go negative on capacity violation.
        self.remaining_vms[action] -= svc.requirement
        self.ecu_placements[action].add(self._step)
        self._update_ecu_allowed(action, self._step)
        if ru > 0:
            self._total_ru += ru
        _active = sum(1 for j in range(self.N) if self.ecu_placements[j])
        self.ar = self._total_ru / _active if _active > 0 else 0.0
        self._step += 1

        done = self._step >= self.M
        total_viol = self.capacity_violations + self.conflict_violations
        return self._obs(), float(ru + cap_penalty + conflict_penalty), done, False, {
            "ar":                  self.ar,
            "step":                self._step,
            "services_placed":     self._step,
            "cap_violated":        cap_violated,
            "conflict_violated":   conflict_violated,
            "capacity_violations": self.capacity_violations,
            "conflict_violations": self.conflict_violations,
            "total_violations":    total_viol,
            "violation_rate":      total_viol / self._step,
        }

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs "
                  f"| AR={self.ar:.4f} | cap_viol={self.capacity_violations} "
                  f"conflict_viol={self.conflict_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f} | cap_viol={self.capacity_violations} "
                  f"conflict_viol={self.conflict_violations}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random as rng
    rng.seed(42)
    N, M = 7, 10
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(rng.sample(range(50, 200, 5), N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(rng.sample(range(10, 80, 5), M))]

    print("ECU capacities :", [e.capacity for e in ecus])
    print("Service demands:", [s.requirement for s in services])

    env = DDQNEnv(ecus, services)
    obs, _ = env.reset()
    print(f"\nObs shape : {obs.shape}  (expected {5 * N + 6 + M})")

    print("\n── Valid greedy policy run ──")
    done = False
    while not done:
        svc = env.services[env._step]
        valid = [j for j in range(N) if env.remaining_vms[j] >= svc.requirement]
        a = valid[0] if valid else int(np.argmax(env.remaining_vms))
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR             : {info['ar']:.4f}")
    print(f"Services placed      : {info['services_placed']}/{M}")
    print(f"Capacity violations  : {info['capacity_violations']}")
    print(f"Conflict violations  : {info['conflict_violations']}")
