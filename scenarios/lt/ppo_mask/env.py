"""
P4 Environment — Hard capacity AND hard conflict (both masked).

N < M: each ECU hosts multiple services.

Constraints:
    - Capacity violation → HARD (action masking strictly prevents selection).
    - Conflict violation → HARD (action masking strictly prevents selection).
    If no valid ECU exists, action_masks() returns all-False.
    During evaluation, evaluate_model() breaks the episode cleanly (zero violations,
    services_placed < M). During training, MaskablePPO samples uniformly and incurs
    a heavy penalty, learning to avoid infeasible states.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from ilp.objects import ECU, SVC


class P4Env(gym.Env):
    """
    Each episode assigns M services to N ECUs (N < M), one service per step.
    Multiple services share ECUs.

    action_masks() returns True only for ECUs that satisfy BOTH capacity
    and conflict constraints.  If no such ECU exists, the fallback ECU
    with maximum remaining capacity is used (heavy penalty applied).

    Observation (shape: 5N+7+2M):
        [0]          current service demand (normalised)
        [1]          current cumulative AR
        [2]          sum of remaining ECU capacity (normalised, clipped ≥ 0)
        [3]          sum of remaining service demand (normalised)
        [4]          fraction of ECUs with sufficient capacity for current service
        [5]          fraction of services remaining
        [6]          bottleneck risk: mean of 1/(valid_ecu_count+1) over remaining
                     services (incl. current) -- a continuous aggregate dead-end-
                     proximity signal (see P4Env._bottleneck_risk()) the policy
                     previously had to infer itself from the raw per-service
                     valid-ECU counts below.
        [7:7+N]      initial capacity fraction per ECU
        [7+N:7+2N]   remaining capacity fraction per ECU
        [7+2N:7+3N]  conflict flag per ECU (1 = placing current svc here violates a conflict set)
        [7+3N:7+4N]  ECU allowed fraction (fraction of SVCs still placeable without conflict)
        [7+4N:7+5N]  valid-action flags (1 = capacity OK AND no conflict)
        [7+5N:7+5N+M] remaining service demands (sorted descending)
        [7+5N+M:7+5N+2M] valid ECU count per remaining service (normalised by N; 0 for placed)

    Reward:
        terminal: M*ar (zero violations) or -M (any violation), v2.2.0-style
        + potential-based shaping F(s,a,s') = gamma*Phi(s') - Phi(s) every
          step (incl. terminal), Phi(s) = -bottleneck_shaping_weight *
          bottleneck_risk(s). Provably does not change the optimal policy
          for any bottleneck_shaping_weight >= 0 (Ng, Harada & Russell 1999);
          weight=0.0 (the default) makes shaping an exact no-op.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, ecus: list[ECU], services: list[SVC], scenarios=None,
        bottleneck_shaping_weight: float = 0.0, gamma: float = 0.99,
    ):
        super().__init__()
        self._scenarios = scenarios
        self.ecus     = ecus
        self.services = services
        self.N = len(ecus)
        self.M = len(services)
        # Potential-based reward shaping (Ng, Harada & Russell 1999):
        # F(s,a,s') = gamma*Phi(s') - Phi(s), Phi(s) = -beta*bottleneck_risk(s).
        # Guaranteed not to change the optimal policy for ANY beta>=0 (unlike
        # ad-hoc dense shaping, which is why v1.1.0 dropped dense reward in
        # favour of pure sparse) -- it only redistributes the terminal -M/M*ar
        # signal earlier, giving PPO a per-step hint about whether the action
        # just taken made a future dead-end more or less likely, instead of
        # only finding out M-minus-however-many steps later. beta=0.0 is a
        # strict no-op (F=0 identically), so existing callers are unaffected.
        self._shaping_beta = float(bottleneck_shaping_weight)
        self._shaping_gamma = float(gamma)

        self.action_space = gym.spaces.Discrete(self.N)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5 * self.N + 7 + 2 * self.M,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms:   np.ndarray
        self.ecu_placements:  list[set]
        self.conflict_sets:   list[set]
        self.ecu_allowed:     list[set]
        self.ar:              float
        self._step:           int
        self.capacity_violations:  int
        self.conflict_violations:  int
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

    def _bottleneck_risk(self) -> float:
        """Aggregate dead-end-proximity signal over not-yet-placed services
        (self._step onward): mean of 1/(valid_ecu_count+1) across them.

        v2.4.0 originally used "fraction of remaining services with <=1
        valid ECU" -- a hard threshold that jumps discontinuously the moment
        a service's valid_ecu_count crosses from 2 to 1, which likely
        contributed to the noisy/negative potential-shaping ablation results
        (fixed beta=2.0 and annealed both underperformed beta=0.0). This
        continuous version changes smoothly as any remaining service's
        option count changes by 1 (1/(n+1) term shrinks smoothly as n grows:
        1.0 at n=0 "dead", 0.5 at n=1, ~0.09 at n=10), instead of only
        registering something once a service is already down to its last
        option. Still 0.0 exactly at/after termination (self._step>=self.M),
        preserving Ng et al. 1999's "absorbing-state potential = 0"
        requirement for the shaping's policy-invariance guarantee."""
        if self._step >= self.M:
            return 0.0
        n_remaining = max(self.M - self._step, 1)
        risk_sum = 0.0
        for i in range(self._step, self.M):
            n_valid = sum(
                1 for j in range(self.N)
                if self.remaining_vms[j] >= self.services[i].requirement
                and not self._has_conflict(j, i)
            )
            risk_sum += 1.0 / (n_valid + 1)
        return risk_sum / n_remaining

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs, _cs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
            self.conflict_sets = [set(cs) for cs in _cs]
        else:
            self.conflict_sets = self._init_conflict_sets()
        # Sort services descending and remap conflict_set indices to match the new order.
        sort_idx = sorted(range(self.M), key=lambda i: -self.services[i].requirement)
        self.services = [self.services[i] for i in sort_idx]
        inv_perm = [0] * self.M
        for new_i, old_i in enumerate(sort_idx):
            inv_perm[old_i] = new_i
        self.conflict_sets = [{inv_perm[k] for k in cs} for cs in self.conflict_sets]
        self.remaining_vms   = self.initial_vms.copy()
        self.ecu_placements  = [set() for _ in range(self.N)]
        self._req_arr = np.array([s.requirement for s in self.services], dtype=np.float32)
        self._n_active = 0
        self.ecu_allowed     = [set(range(self.M)) for _ in range(self.N)]
        self.ar              = 0.0
        self._total_ru       = 0.0
        self._step           = 0
        self.capacity_violations = 0
        self.conflict_violations = 0
        self.valid_placed = 0
        self.episode_has_cap_violation      = False
        self.episode_has_conflict_violation = False
        return self._obs(), {}

    # ── action mask (capacity AND conflict) ──────────────────────────────────
    def action_masks(self) -> np.ndarray:
        if self._step >= self.M:
            return np.zeros(self.N, dtype=bool)
        svc = self.services[self._step]
        mask = np.array(
            [(self.remaining_vms[j] >= svc.requirement) and (not self._has_conflict(j, self._step))
             for j in range(self.N)],
            dtype=bool,
        )
        return mask

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
                float(np.sum(self._req_arr[self._step:])) / total_cap
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
        remaining_svcs = np.zeros(self.M, dtype=np.float32)
        if self._step < self.M:
            remaining_svcs[self._step:] = self._req_arr[self._step:] / max_cap

        ecu_allowed_frac = np.array(
            [len(self.ecu_allowed[j]) / self.M for j in range(self.N)],
            dtype=np.float32,
        )

        svc_valid_ecus = np.zeros(self.M, dtype=np.float32)
        for i in range(self._step, self.M):
            n_valid = sum(
                1 for j in range(self.N)
                if self.remaining_vms[j] >= self.services[i].requirement
                and not self._has_conflict(j, i)
            )
            svc_valid_ecus[i] = n_valid / self.N
        # Delegates to _bottleneck_risk() rather than re-deriving inline, so
        # the observation feature and step()'s shaping potential can never
        # drift apart (they did briefly during v2.4.0 development).
        bottleneck_risk = np.float32(self._bottleneck_risk())

        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            np.array([remaining_usable_capacity_sum], dtype=np.float32),
            np.array([remaining_service_demand_sum], dtype=np.float32),
            np.array([remaining_usable_ecu_count], dtype=np.float32),
            np.array([remaining_services_count], dtype=np.float32),
            np.array([bottleneck_risk], dtype=np.float32),
            initial_cap_pct,
            remaining_abs_norm,
            conflict_flag,
            ecu_allowed_frac,
            valid_flag,
            remaining_svcs,
            svc_valid_ecus,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert 0 <= action < self.N, f"Invalid action {action}"
        svc = self.services[self._step]

        cap_violated      = bool(self.remaining_vms[action] < svc.requirement)
        conflict_violated = self._has_conflict(action, self._step)

        if cap_violated:
            self.capacity_violations += 1
            self.episode_has_cap_violation = True
        if conflict_violated:
            self.conflict_violations += 1
            self.episode_has_conflict_violation = True
        if not (cap_violated or conflict_violated):
            self.valid_placed += 1

        violated = cap_violated or conflict_violated
        # Heavy penalty applied only when forced-overflow fallback triggers a violation.
        violation_penalty = -2.0 if violated else 0.0
        ru = 0.0 if violated else svc.requirement / (self.initial_vms[action] + 1e-8)

        # Phi(s) BEFORE this transition's mutations -- see step()'s tail for
        # Phi(s') and the shaping term itself.
        phi_s = -self._shaping_beta * self._bottleneck_risk()

        self.remaining_vms[action] -= svc.requirement
        _was_empty = not self.ecu_placements[action]
        self.ecu_placements[action].add(self._step)
        if _was_empty:
            self._n_active += 1
        self._update_ecu_allowed(action, self._step)
        if ru > 0:
            self._total_ru += ru
        _active = self._n_active
        self.ar = self._total_ru / _active if _active > 0 else 0.0
        self._step += 1

        done = self._step >= self.M
        total_viol = self.capacity_violations + self.conflict_violations
        if done:
            # AR is now the thing being optimized, not just observed: a
            # zero-violation episode scores M*AR (quality-proportional)
            # instead of a flat +M, so PPO's gradient actually distinguishes
            # a 0.55-AR success from a 0.90-AR success — previously both
            # scored the same +M and the policy had no signal to prefer one
            # over the other. Violations remain a hard constraint: -M is
            # strictly worse than any success (M*AR <= M since AR in (0,1]),
            # so the ordering "any valid placement beats any violation" is
            # preserved regardless of how low that placement's AR is.
            reward = float(self.M) * self.ar if total_viol == 0 else -float(self.M)
        else:
            reward = 0.0

        # Potential-based shaping F(s,a,s') = gamma*Phi(s') - Phi(s), added on
        # top of the terminal/zero reward above. beta=0.0 makes phi_s and
        # phi_s_next both identically 0.0, so shaping is an exact no-op --
        # this line is always safe to leave in regardless of _shaping_beta.
        # Phi(terminal) = -beta*_bottleneck_risk() = -beta*0.0 = 0.0
        # (the self._step >= self.M branch), satisfying the Ng et al. 1999
        # requirement that the absorbing state's potential be zero, so the
        # optimal-policy-invariance guarantee holds exactly at any beta>=0.
        phi_s_next = -self._shaping_beta * self._bottleneck_risk()
        shaping = self._shaping_gamma * phi_s_next - phi_s
        reward += shaping

        step_reward = ru / max(_active, 1)
        info = {
            "ar":                  self.ar,
            "step":                self._step,
            "services_placed":     self._step,
            "valid_placed":        self.valid_placed,
            "ecus_used":          _active,
            "capacity_violations": self.capacity_violations,
            "conflict_violations": self.conflict_violations,
            "total_violations":    total_viol,
            "violation_rate":      total_viol / self._step,
            "episode_has_cap_violation":      self.episode_has_cap_violation,
            "episode_has_conflict_violation": self.episode_has_conflict_violation,
        }
        return self._obs(), reward, done, False, info

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            valid = int(np.sum(self.action_masks()))
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs "
                  f"| AR={self.ar:.4f} | valid ECUs={valid}/{self.N} "
                  f"| cap_viol={self.capacity_violations} conflict_viol={self.conflict_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f} "
                  f"| cap_viol={self.capacity_violations} conflict_viol={self.conflict_violations}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    N, M = 7, 10
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(random.sample(range(50, 200, 5), N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(random.sample(range(10, 80, 5), M))]

    print("ECU capacities :", [e.capacity for e in ecus])
    print("Service demands:", [s.requirement for s in services])

    env = P4Env(ecus, services)
    obs, _ = env.reset()
    print(f"\nObs shape : {obs.shape}  (expected {5 * N + 7 + 2 * M})")

    print("\n── Valid action policy run ──")
    done = False
    while not done:
        mask  = env.action_masks()
        valid = np.where(mask)[0]
        a = int(valid[0])
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR             : {info['ar']:.4f}")
    print(f"Services placed      : {info['services_placed']}/{M}")
    print(f"Capacity violations  : {info['capacity_violations']}")
    print(f"Conflict violations  : {info['conflict_violations']}")
