"""
P6 Environment — RL WITH best-fit patch optimization.

Design intent (matches docs.md):
    • When assigning to an already-occupied ECU, a best-fit patch algorithm
        is applied to relocate one service to a better-fitting free ECU.
    • Episode NEVER terminates early — always runs M steps.
    • Reward equals the exact increase in total utilisation after each action.
    • Observation explicitly includes the currently loaded fraction of each ECU.
    • Scenarios: reset() randomly picks one of the provided scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_ilp.objects import ECU, SVC


# ─────────────────────────────────────────────────────────────────────────────
#  P6 Environment — Best-fit patch optimization
# ─────────────────────────────────────────────────────────────────────────────

class P6Env(gym.Env):
    """
    Each episode assigns M services to N ECUs, one service per step.

    When the selected ECU is already occupied, a best-fit patch algorithm
    is applied: one service is relocated to the tightest-fitting free ECU
    to minimise waste and reduce constraint violations.

        Reward:
            • increase in total utilisation after this action

        Observation  (shape: 2N+2):
      [0]   current service demand, normalised by max initial capacity
      [1]   current cumulative AR
            [2:2+N]      remaining capacity fraction per ECU
            [2+N:2+2N]   current assigned load fraction per ECU
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

        # remaining capacity can go below 0; load fraction can exceed 1
        self.observation_space = gym.spaces.Box(
            low  = -1.0,
            high =  2.0,
            shape= (2 * self.N + 2,),
            dtype= np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms: np.ndarray
        self.ecu_loads:     np.ndarray
        self.ecu_assigned:  np.ndarray
        self.ar:            float
        self._step:         int
        self.reset()

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
        self.remaining_vms  = self.initial_vms.copy()
        self.ecu_loads      = np.zeros(self.N, dtype=np.float32)
        self.ecu_assigned   = np.zeros(self.N, dtype=bool)
        self.ecu_service_idx = np.full(self.N, -1, dtype=int)  # ECU -> service index (-1 = free)
        self.ar             = 0.0
        self._step          = 0
        self.capacity_violations       = 0
        self.single_service_violations = 0
        return self._obs(), {}

    # ── observation ──────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        if self._step >= self.M:
            service_demand_norm = np.float32(0.0)
        else:
            svc = self.services[self._step]
            service_demand_norm = np.float32(svc.requirement / (np.max(self.initial_vms) + 1e-8))

        remaining_pct = self.remaining_vms / (self.initial_vms + np.float32(1e-8))
        load_pct = self.ecu_loads / (self.initial_vms + np.float32(1e-8))

        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            remaining_pct,
            load_pct,
        ]).astype(np.float32)

    def _compute_total_util(self) -> float:
        active_mask = self.ecu_assigned.astype(bool)
        if not np.any(active_mask):
            return 0.0
        return float(np.sum(self.ecu_loads[active_mask] / (self.initial_vms[active_mask] + 1e-8)))

    def _compute_ar(self) -> float:
        active_ecus = int(np.sum(self.ecu_assigned))
        if active_ecus == 0:
            return 0.0
        return self._compute_total_util() / active_ecus

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert 0 <= action < self.N, f"Invalid action {action}"
        svc = self.services[self._step]
        prev_total_util = self._compute_total_util()

        if self.ecu_assigned[action]:
            # ── Best-fit patch algorithm (see patch_algorithm_en.md) ──────────
            ecu_n = action
            s1    = self.services[self.ecu_service_idx[ecu_n]]
            s2    = svc
            cap_n = float(self.initial_vms[ecu_n])

            # Step 1: determine s_stay / s_move
            if s2.requirement > s1.requirement:
                s_stay, s_move = s2, s1
                if cap_n < s2.requirement:
                    self.capacity_violations += 1
            else:
                s_stay, s_move = s1, s2

            # Step 2: find best-fit ecu_m (capacity sufficient, tightest fit)
            best_ecu_m    = -1
            best_remaining = float('inf')
            for j in range(self.N):
                if not self.ecu_assigned[j] and j != ecu_n:
                    if self.initial_vms[j] >= s_move.requirement:
                        leftover = float(self.remaining_vms[j]) - s_move.requirement
                        if leftover < best_remaining:
                            best_remaining = leftover
                            best_ecu_m     = j

            if best_ecu_m != -1:
                # Step 3: execute relocation (normal path)
                ecu_m = best_ecu_m
                cap_m = float(self.initial_vms[ecu_m])
                self.ecu_loads[ecu_n]        = float(s_stay.requirement)
                self.ecu_loads[ecu_m]        = float(s_move.requirement)
                self.remaining_vms[ecu_n]    = cap_n - s_stay.requirement
                self.remaining_vms[ecu_m]    = cap_m - s_move.requirement
                self.ecu_service_idx[ecu_n]  = self.services.index(s_stay)
                self.ecu_service_idx[ecu_m]  = self.services.index(s_move)
                self.ecu_assigned[ecu_m]     = True

            else:
                # Step 5: fallback — enumerate all free ECUs, both arrangements
                free_ecus = [j for j in range(self.N)
                             if not self.ecu_assigned[j] and j != ecu_n]
                if not free_ecus:
                    # No relocation target exists: keep original placement on ecu_n
                    # and count this action as one duplicate-selection violation.
                    self.single_service_violations += 1
                    self._step += 1
                    self.ar = self._compute_ar()
                    done = self._step >= self.M
                    reward = self._compute_total_util() - prev_total_util
                    total_viol = self.capacity_violations + self.single_service_violations
                    info = {
                        "ar":   self.ar,
                        "step": self._step,
                        "services_placed":         self._step,
                        "capacity_violations":       self.capacity_violations,
                        "single_service_violations": self.single_service_violations,
                        "total_violations":          total_viol,
                        "violation_rate":            total_viol / self._step,
                    }
                    return self._obs(), reward, done, False, info

                best_ecu_m   = -1
                best_viol    = 3        # sentinel > 2
                best_score   = -1.0
                best_s_on_n  = s1
                best_s_on_m  = s2

                for j in free_ecus:
                    cap_m = float(self.initial_vms[j])
                    for (s_on_n, s_on_m) in [(s1, s2), (s2, s1)]:
                        viol = 0
                        if s_on_n.requirement > cap_n: viol += 1
                        if s_on_m.requirement > cap_m: viol += 1
                        if viol >= 2:                  continue   # never allow both violated
                        score = (s1.requirement + s2.requirement) / (cap_n + cap_m + 1e-8)
                        if viol < best_viol or (viol == best_viol and score > best_score):
                            best_viol   = viol
                            best_score  = score
                            best_ecu_m  = j
                            best_s_on_n = s_on_n
                            best_s_on_m = s_on_m

                ecu_m = best_ecu_m
                cap_m = float(self.initial_vms[ecu_m])
                self.capacity_violations        += best_viol
                self.ecu_loads[ecu_n]            = float(best_s_on_n.requirement)
                self.ecu_loads[ecu_m]            = float(best_s_on_m.requirement)
                self.remaining_vms[ecu_n]        = cap_n - best_s_on_n.requirement
                self.remaining_vms[ecu_m]        = cap_m - best_s_on_m.requirement
                self.ecu_service_idx[ecu_n]      = self.services.index(best_s_on_n)
                self.ecu_service_idx[ecu_m]      = self.services.index(best_s_on_m)
                self.ecu_assigned[ecu_m]         = True
            self._step += 1

        else:
            # ── Normal assignment (ecu_n was free) ────────────────────────────
            # Constraint check: capacity violation
            if self.remaining_vms[action] < svc.requirement:
                self.capacity_violations += 1
            self.ecu_loads[action]        = float(svc.requirement)
            self.remaining_vms[action]   -= svc.requirement
            self.ecu_assigned[action]     = True
            self.ecu_service_idx[action]  = self._step   # record service index on this ECU
            self._step += 1

        self.ar = self._compute_ar()
        done   = self._step >= self.M
        reward = self._compute_total_util() - prev_total_util

        total_viol = self.capacity_violations + self.single_service_violations
        info = {
            "ar":   self.ar,
            "step": self._step,
            "services_placed":         self._step,
            "capacity_violations":       self.capacity_violations,
            "single_service_violations": self.single_service_violations,
            "total_violations":          total_viol,
            "violation_rate":            total_viol / self._step,
        }
        return self._obs(), reward, done, False, info

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | service needs {svc.requirement} VMs | AR={self.ar:.4f} "
                  f"| cap_viol={self.capacity_violations} dup_viol={self.single_service_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f} "
                  f"| cap_viol={self.capacity_violations} dup_viol={self.single_service_violations}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)
    N, M = 5, 8
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(random.sample(range(50, 200, 5), N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(random.sample(range(10, 100, 5), M))]

    print("ECU capacities :", [e.capacity for e in ecus])
    print("Service demands:", [s.requirement for s in services])

    env = P6Env(ecus, services)
    obs, _ = env.reset()
    print(f"\nObs shape : {obs.shape}  (expected {N+2})")
    print(f"Obs       : {obs}")

    print("\n── Random policy run ──")
    done = False
    while not done:
        a = env.action_space.sample()
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR              : {info['ar']:.4f}")
    print(f"Capacity violations   : {info['capacity_violations']}")
    print(f"Dup-ECU violations    : {info['single_service_violations']}")
    print(f"Total violations      : {info['total_violations']}")
    print(f"Violation rate        : {info['violation_rate']:.2%}")
