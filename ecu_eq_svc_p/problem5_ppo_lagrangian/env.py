"""
P5 Environment — Lagrangian Constraint Relaxation.

Design:
    - Capacity violation  → fixed large penalty (-2.0 per step); episode continues.
    - Conflict violation  → adaptive Lagrangian penalty (λ + base_penalty) * c_t;
                            λ is updated externally via dual ascent.
    - Episode always runs M steps; remaining_vms can go negative.

Reward (potential-based shaping, same as P4):
    Δar   = ar_new - ar_prev          (dense utilisation signal)
    - cap_penalty                      (-2.0 if capacity violated, else 0)
    - (λ + base_penalty) * c_t        (Lagrangian conflict penalty)
    + ar_final  (terminal, no-violation episodes only)

Services are sorted descending by requirement at each reset (FFD order).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_ilp.objects import ECU, SVC


class LagrangeEnv(gym.Env):
    """
    Observation (shape: 4N+7+M):
        [0]          current service demand (normalised)
        [1]          current cumulative AR
        [2]          sum of remaining ECU capacity (normalised, clipped ≥ 0)
        [3]          sum of remaining service demand (normalised)
        [4]          fraction of ECUs with sufficient capacity
        [5]          fraction of services remaining
        [6:6+N]      initial capacity fraction per ECU
        [6+N:6+2N]   remaining capacity fraction per ECU (can be negative)
        [6+2N:6+3N]  conflict flag per ECU
        [6+3N:6+4N]  valid-action flags (1 = sufficient capacity)
        [6+4N:6+4N+M] remaining service demands (sorted descending)
        [6+4N+M]     current λ value, normalised by λ_max

    Reward per step:
        r_t = match_gain - (lambda_val + base_penalty) * c_t + terminal_bonus
    """

    metadata = {"render_modes": []}

    def __init__(self, ecus: list[ECU], services: list[SVC],
                 scenarios=None, lambda_init: float = 0.0,
                 lambda_max: float = 10.0):
        super().__init__()
        self._scenarios = scenarios
        self.ecus       = ecus
        self.services   = services
        self.N          = len(ecus)
        self.M          = len(services)
        self.lambda_val = float(lambda_init)
        self.lambda_max = max(float(lambda_max), 1e-8)

        self.action_space = gym.spaces.Discrete(self.N)
        # remaining capacity can go negative due to soft capacity constraint
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=1.0, shape=(4 * self.N + 7 + 2 * self.M,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms:       np.ndarray
        self.ecu_placements:      list[set]
        self.conflict_sets:       list[set]
        self.ar:                  float
        self._step:               int
        self.episode_violations:  int
        self.cap_violations:      int
        self.conflict_violations: int
        self.reset()

    def set_lambda(self, val: float):
        self.lambda_val = float(val)

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

    def action_masks(self) -> np.ndarray:
        """Hard capacity mask: True = ECU has enough remaining capacity."""
        if self._step >= self.M:
            return np.zeros(self.N, dtype=bool)
        svc = self.services[self._step]
        mask = self.remaining_vms >= svc.requirement
        if not np.any(mask):
            return np.ones(self.N, dtype=bool)  # unavoidable infeasibility
        return mask

    # ── reset ─────────────────────────────────────────────────────────────────
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

        # Sort services descending by requirement (FFD order).
        # Remap conflict_set indices so self._step always indexes sorted services.
        sort_idx = sorted(range(self.M), key=lambda i: -self.services[i].requirement)
        self.services = [self.services[i] for i in sort_idx]
        inv_perm = [0] * self.M
        for new_i, old_i in enumerate(sort_idx):
            inv_perm[old_i] = new_i
        self.conflict_sets = [{inv_perm[k] for k in cs} for cs in self.conflict_sets]

        self.remaining_vms   = self.initial_vms.copy()
        self.ecu_placements  = [set() for _ in range(self.N)]
        self.ecu_allowed     = [set(range(self.M)) for _ in range(self.N)]
        self._req_arr = np.array([s.requirement for s in self.services], dtype=np.float32)
        self._n_active = 0
        self.ar              = 0.0
        self._total_ru       = 0.0
        self._step           = 0
        self.episode_violations  = 0
        self.cap_violations      = 0
        self.conflict_violations = 0
        self.valid_placed = 0
        self.episode_has_cap_violation      = False
        self.episode_has_conflict_violation = False
        return self._obs(), {}

    # ── observation ───────────────────────────────────────────────────────────
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
                float(np.sum(self._req_arr[self._step:])) / total_cap
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
        remaining_svcs = np.zeros(self.M, dtype=np.float32)
        if self._step < self.M:
            remaining_svcs[self._step:] = self._req_arr[self._step:] / max_cap
        lambda_norm = np.float32(self.lambda_val / self.lambda_max)

        svc_valid_ecus = np.zeros(self.M, dtype=np.float32)
        for i in range(self._step, self.M):
            svc_valid_ecus[i] = sum(
                1 for j in range(self.N)
                if self.remaining_vms[j] >= self.services[i].requirement
                and not self._has_conflict(j, i)
            ) / self.N

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
            svc_valid_ecus,
            np.array([lambda_norm], dtype=np.float32),
        ]).astype(np.float32)

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        svc = self.services[self._step]
        ar_prev = self.ar  # capture before placement for Δar reward

        cap_violated      = bool(self.remaining_vms[action] < svc.requirement)
        conflict_violated = self._has_conflict(action, self._step)
        # λ (dual variable) tracks CONFLICT only; capacity gets a fixed large penalty.
        c_t = 1.0 if conflict_violated else 0.0

        ru = svc.requirement / (self.initial_vms[action] + 1e-8)
        self.remaining_vms[action] -= svc.requirement
        _was_empty = not self.ecu_placements[action]
        self.ecu_placements[action].add(self._step)
        if _was_empty:
            self._n_active += 1
        self._update_ecu_allowed(action, self._step)
        self._total_ru += ru
        _active = self._n_active
        self.ar = self._total_ru / _active
        self._step += 1

        if cap_violated:
            self.cap_violations += 1
            self.episode_has_cap_violation = True
            self.episode_violations += 1
        if conflict_violated:
            self.conflict_violations += 1
            self.episode_has_conflict_violation = True
            self.episode_violations += 1
        if not (cap_violated or conflict_violated):
            self.valid_placed += 1

        done = self._step >= self.M

        # Reward:
        #   Δar            — potential-based dense utilisation signal
        #   cap_penalty    — fixed -2.0 for capacity overflow (explicit, consistent gradient)
        #   Lagrangian     — adaptive conflict penalty via dual ascent
        #   terminal_bonus — +ar_final for clean episodes; 0 otherwise (avoid -ar instability)
        delta_ar       = self.ar - ar_prev
        cap_penalty    = -2.0 if cap_violated else 0.0
        base_penalty   = 0.2
        terminal_bonus = self.ar if (done and self.episode_violations == 0) else 0.0
        reward = float(delta_ar + cap_penalty
                       - (self.lambda_val + base_penalty) * c_t
                       + terminal_bonus)

        return self._obs(), reward, done, False, {
            "ar":                             self.ar,
            "violated":                       bool(cap_violated or conflict_violated),
            "violations_ep":                  self.episode_violations,
            "viol_rate_ep":                   self.episode_violations / self._step,
            # λ should track ONLY conflict violations; cap violations are handled separately
            "conf_viol_rate_ep":              self.conflict_violations / self._step,
            "cap_violations":                 self.cap_violations,
            "conflict_violations":            self.conflict_violations,
            "services_placed":                self._step,
            "valid_placed":                   self.valid_placed,
            "ecus_used":                      _active,
            "lambda":                         self.lambda_val,
            "episode_has_cap_violation":      self.episode_has_cap_violation,
            "episode_has_conflict_violation": self.episode_has_conflict_violation,
        }

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs"
                  f" | AR={self.ar:.4f} | λ={self.lambda_val:.3f}"
                  f" | violations={self.episode_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f} | violations={self.episode_violations}/{self.M}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    N, M = 7, 4
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(random.sample(range(50, 200, 5), N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(random.sample(range(10, 100, 5), M))]

    env = LagrangeEnv(ecus, services, lambda_init=1.0)
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}  (expected {4*N + 7 + M})")

    done = False
    while not done:
        a = env.action_space.sample()
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR: {info['ar']:.4f} | violations: {info['violations_ep']}")
