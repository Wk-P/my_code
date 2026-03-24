"""
P6 Environment — RL WITH best-fit patch optimization.

Design intent (matches docs.md):
    • When assigning to an already-occupied ECU, a best-fit patch algorithm
        is applied only if it can keep the allocation fully feasible.
    • If the selected action cannot be repaired without violations, the episode
        terminates immediately with an unfinished-demand penalty.
    • Reward equals the exact increase in total utilisation after each action,
        plus an early-stop penalty when no feasible continuation exists.
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
            • early-stop penalty if a zero-violation placement/patch is impossible

                Observation (shape: 4N+6+M):
            Standard PPO state from thinking.md:
            [0]                current service demand
            [1]                current cumulative AR
            [2]                sum of remaining usable ECU capacity
            [3]                sum of remaining service demand
            [4]                number of remaining usable ECUs
            [5]                number of remaining services
            [6:6+N]            initial capacity fraction per ECU
            [6+N:6+2N]         remaining capacity fraction per ECU
            [6+2N:6+3N]        ECU occupied flags
            [6+3N:6+4N]        valid-action flags for current service
            [6+4N:6+4N+M]      remaining service demands (sorted descending)
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

        # Under strict feasibility all observation terms stay within [0, 1].
        self.observation_space = gym.spaces.Box(
            low  = 0.0,
            high =  1.0,
            shape= (4 * self.N + 6 + self.M,),
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
        self.services = sorted(self.services, key=lambda s: s.requirement, reverse=True)
        self.remaining_vms  = self.initial_vms.copy()
        self.ecu_loads      = np.zeros(self.N, dtype=np.float32)
        self.ecu_assigned   = np.zeros(self.N, dtype=bool)
        self.ecu_service_idx = np.full(self.N, -1, dtype=int)  # ECU -> service index (-1 = free)
        self.ar             = 0.0
        self._step          = 0
        self.capacity_violations       = 0
        self.single_service_violations = 0
        return self._obs(), {}

    def _has_feasible_patch(self, ecu_n: int, svc: SVC) -> bool:
        if not self.ecu_assigned[ecu_n]:
            return bool(self.remaining_vms[ecu_n] >= svc.requirement)

        s1 = self.services[self.ecu_service_idx[ecu_n]]
        cap_n = float(self.initial_vms[ecu_n])
        for j in range(self.N):
            if self.ecu_assigned[j] or j == ecu_n:
                continue
            cap_m = float(self.initial_vms[j])
            for s_on_n, s_on_m in ((s1, svc), (svc, s1)):
                if s_on_n.requirement <= cap_n and s_on_m.requirement <= cap_m:
                    return True
        return False

    def action_masks(self) -> np.ndarray:
        if self._step >= self.M:
            return np.zeros(self.N, dtype=bool)
        svc = self.services[self._step]
        return np.array([self._has_feasible_patch(j, svc) for j in range(self.N)], dtype=bool)

    # ── observation ──────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        if self._step >= self.M:
            service_demand_norm = np.float32(0.0)
        else:
            svc = self.services[self._step]
            service_demand_norm = np.float32(svc.requirement / (np.max(self.initial_vms) + 1e-8))

        remaining_pct = self.remaining_vms / (self.initial_vms + np.float32(1e-8))
        initial_cap_pct = self.initial_vms / (np.max(self.initial_vms) + np.float32(1e-8))
        occupied_flag = self.ecu_assigned.astype(np.float32)
        if self._step >= self.M:
            valid_flag = np.zeros(self.N, dtype=np.float32)
        else:
            valid_flag = self.action_masks().astype(np.float32)

        total_cap = float(np.sum(self.initial_vms) + 1e-8)
        max_cap = float(np.max(self.initial_vms) + 1e-8)
        usable_mask = ~self.ecu_assigned
        remaining_usable_capacity_sum = np.float32(
            np.sum(self.remaining_vms[usable_mask]) / total_cap
        )
        if self._step >= self.M:
            remaining_service_demand_sum = np.float32(0.0)
            remaining_services_count = np.float32(0.0)
        else:
            remaining_service_demand_sum = np.float32(
                sum(self.services[t].requirement for t in range(self._step, self.M)) / total_cap
            )
            remaining_services_count = np.float32((self.M - self._step) / max(self.M, 1))
        remaining_usable_ecu_count = np.float32(np.sum(usable_mask) / max(self.N, 1))
        remaining_svcs = np.array(
            [self.services[t].requirement / max_cap if t >= self._step else 0.0 for t in range(self.M)],
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
            remaining_pct,
            occupied_flag,
            valid_flag,
            remaining_svcs,
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
            # ── Best-fit patch algorithm under strict feasibility ─────────────
            ecu_n = action
            s1    = self.services[self.ecu_service_idx[ecu_n]]
            s2    = svc
            cap_n = float(self.initial_vms[ecu_n])

            free_ecus = [j for j in range(self.N) if not self.ecu_assigned[j] and j != ecu_n]
            best_ecu_m  = -1
            best_score  = -1.0
            best_s_on_n = s1
            best_s_on_m = s2

            for j in free_ecus:
                cap_m = float(self.initial_vms[j])
                for (s_on_n, s_on_m) in [(s1, s2), (s2, s1)]:
                    if s_on_n.requirement > cap_n or s_on_m.requirement > cap_m:
                        continue
                    score = (s1.requirement + s2.requirement) / (cap_n + cap_m + 1e-8)
                    if score > best_score:
                        best_score  = score
                        best_ecu_m  = j
                        best_s_on_n = s_on_n
                        best_s_on_m = s_on_m

            if best_ecu_m == -1:
                self.single_service_violations += 1
                remaining_services = self.M - self._step
                unplaced_demand = sum(self.services[i].requirement for i in range(self._step, self.M))
                penalty = -float(unplaced_demand) / (np.sum(self.initial_vms) + 1e-8)
                return self._obs(), penalty, True, False, {
                    "ar": self.ar,
                    "step": self._step,
                    "services_placed": self._step,
                    "capacity_violations": self.capacity_violations,
                    "single_service_violations": self.single_service_violations,
                    "total_violations": self.capacity_violations + self.single_service_violations,
                    "violation_rate": 1.0 if remaining_services > 0 else 0.0,
                }

            ecu_m = best_ecu_m
            cap_m = float(self.initial_vms[ecu_m])
            self.ecu_loads[ecu_n]        = float(best_s_on_n.requirement)
            self.ecu_loads[ecu_m]        = float(best_s_on_m.requirement)
            self.remaining_vms[ecu_n]    = cap_n - best_s_on_n.requirement
            self.remaining_vms[ecu_m]    = cap_m - best_s_on_m.requirement
            self.ecu_service_idx[ecu_n]  = self.services.index(best_s_on_n)
            self.ecu_service_idx[ecu_m]  = self.services.index(best_s_on_m)
            self.ecu_assigned[ecu_m]     = True
            self._step += 1

        else:
            # ── Normal assignment (ecu_n was free) ────────────────────────────
            if self.remaining_vms[action] < svc.requirement:
                self.capacity_violations += 1
                remaining_services = self.M - self._step
                unplaced_demand = sum(self.services[i].requirement for i in range(self._step, self.M))
                penalty = -float(unplaced_demand) / (np.sum(self.initial_vms) + 1e-8)
                return self._obs(), penalty, True, False, {
                    "ar": self.ar,
                    "step": self._step,
                    "services_placed": self._step,
                    "capacity_violations": self.capacity_violations,
                    "single_service_violations": self.single_service_violations,
                    "total_violations": self.capacity_violations + self.single_service_violations,
                    "violation_rate": 1.0 if remaining_services > 0 else 0.0,
                }
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
