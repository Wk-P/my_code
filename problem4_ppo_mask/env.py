"""
P4 Environment — RL WITH constraint enforcement via Action Masking.

Key differences from P3:
    - action_masks() returns a boolean array: True = valid ECU, False = invalid.
    - Invalid = capacity insufficient OR ECU already assigned.
    - Agent can ONLY choose from valid actions → 0 constraint violations guaranteed.
    - Reward improved: tight-fit bonus + capacity waste penalty + final AR bonus + demand-based early-stop penalty.
    - If the next service becomes infeasible, the episode terminates early with a penalty
        proportional to the sum of unplaced service demands.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_ilp.objects import ECU, SVC


class P4Env(gym.Env):
    """
    Service Deployment Environment WITH action masking and enhanced reward.

    action_masks() is called by MaskablePPO before each step to know
    which actions are legal. The agent's policy logits for masked actions
    are set to -inf, so they are never selected.

        Observation (shape: 4N+6+M):
      [0]                current service demand, normalised by max ECU capacity
      [1]                current cumulative AR (average resource utilisation)
            [2]                sum of remaining usable ECU capacity (normalised)
            [3]                sum of remaining service demand (normalised)
            [4]                number of remaining usable ECUs (normalised by N)
            [5]                number of remaining services (normalised by M)
            [6:6+N]            initial capacity fraction per ECU
            [6+N:6+2N]         remaining capacity fraction per ECU
            [6+2N:6+3N]        ECU occupied flags (1 = already assigned one service)
            [6+3N:6+4N]        valid-action flags (1 = free AND has enough capacity)
            [6+4N:6+4N+M]      remaining service demands (sorted descending),
                                                 and 0 for already-placed steps

                Reward composition (positive shaping only):
                    reward = match_gain + terminal_bonus
          • match_gain = requirement/capacity for feasible action, else 0.0
          • terminal_bonus = +1.0*AR (clean full episode) or +0.1*AR
          • keep early-stop demand penalty when episode ends before placing all M
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
        # obs: [service_demand, ar,
        #       rem_cap_sum, rem_demand_sum, rem_usable_ecu_cnt, rem_service_cnt,
        #       initial_cap_pct (N), remaining_pct (N), occupied_flag (N),
        #       valid_action_flag (N), remaining_service_demands (M)]
        # total: 4N + 6 + M
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4 * self.N + 6 + self.M,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms: np.ndarray
        self.ecu_assigned:  np.ndarray
        self.ar:            float
        self._step:         int
        self.episode_violations: int
        self.reset()

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
        # Largest-demand-first order helps avoid early infeasibility under masking.
        self.services = sorted(self.services, key=lambda s: s.requirement, reverse=True)
        self.remaining_vms = self.initial_vms.copy()
        self.ecu_assigned  = np.zeros(self.N, dtype=bool)
        self.ar    = 0.0
        self._step = 0
        self.episode_violations = 0
        return self._obs(), {}

    # ── action mask ──────────────────────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        """Return bool array shape (N,). True = action is valid."""
        if self._step >= self.M:
            return np.zeros(self.N, dtype=bool)
        svc = self.services[self._step]
        mask = np.zeros(self.N, dtype=bool)
        for j in range(self.N):
            if (not self.ecu_assigned[j]) and (self.remaining_vms[j] >= svc.requirement):
                mask[j] = True
        return mask

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

        # valid-action flag: true if ECU is free AND has enough capacity for current service
        if self._step >= self.M:
            valid_flag = np.zeros(self.N, dtype=np.float32)
        else:
            svc = self.services[self._step]
            valid_flag = ((~self.ecu_assigned) & (self.remaining_vms >= svc.requirement)).astype(np.float32)

        max_cap = float(np.max(self.initial_vms) + 1e-8)
        total_cap = float(np.sum(self.initial_vms) + 1e-8)

        usable_mask = ~self.ecu_assigned
        remaining_usable_capacity_sum = np.float32(np.sum(np.clip(self.remaining_vms[usable_mask], 0.0, None)) / total_cap)

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
            remaining_pct,
            occupied_flag,
            valid_flag,
            remaining_svcs,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert 0 <= action < self.N, f"Invalid action {action}"
        svc = self.services[self._step]

        # With action masking this should never happen, but defend just in case
        if self.remaining_vms[action] < svc.requirement or self.ecu_assigned[action]:
            self.episode_violations += 1
            match_gain = 0.0
            unplaced_demand = sum(self.services[i].requirement for i in range(self._step, self.M))
            demand_penalty = -float(unplaced_demand) / (np.sum(self.initial_vms) + 1e-8)
            terminal_bonus = 0.1 * self.ar + demand_penalty
            penalty = float(match_gain + terminal_bonus)
            done = True
            return self._obs(), penalty, done, False, {
                "ar": self.ar, "step": self._step,
                "feasible": False,
                "services_placed": self._step,
                "violations_ep": self.episode_violations,
            }

        # ── Valid assignment ──────────────────────────────────────────────────
        ru = svc.requirement / (self.initial_vms[action] + 1e-8)
        self.remaining_vms[action] -= svc.requirement
        self.ecu_assigned[action]   = True

        self.ar = (self.ar * self._step + ru) / (self._step + 1)
        self._step += 1

        done = self._step >= self.M

        # Check if next step would be infeasible (no valid action)
        if not done and not np.any(self.action_masks()):
            done = True   # no valid ECU for next service → early stop

        # ── Reward composition (positive shaping + early-stop penalty) ──────
        match_gain = float(ru)
        terminal_bonus = 0.0

        remaining_services = self.M - self._step
        if done and remaining_services > 0:
            unplaced_demand = sum(self.services[i].requirement for i in range(self._step, self.M))
            demand_penalty = -float(unplaced_demand) / (np.sum(self.initial_vms) + 1e-8)
            terminal_bonus += demand_penalty

        if done:
            if self.episode_violations == 0 and remaining_services == 0:
                terminal_bonus += 1.0 * self.ar
            else:
                terminal_bonus += 0.1 * self.ar

        reward = float(match_gain + terminal_bonus)

        info = {
            "ar":       self.ar,
            "step":     self._step,
            "feasible": True,
            "services_placed": self._step,
            "violations_ep": self.episode_violations,
        }
        return self._obs(), reward, done, False, info

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        mask = self.action_masks()
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs "
                  f"| AR={self.ar:.4f} | valid ECUs={int(np.sum(mask))}/{self.N}")
        else:
            print(f"  Done | AR={self.ar:.4f} | placed {self._step}/{self.M}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random as rng
    rng.seed(42)
    N, M = 5, 4
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(rng.sample(range(50, 200, 5), N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(rng.sample(range(10, 100, 5), M))]

    print("ECU capacities :", [e.capacity for e in ecus])
    print("Service demands:", [s.requirement for s in services])

    env = P4Env(ecus, services)
    obs, _ = env.reset()
    print(f"\nObs shape : {obs.shape}  (expected {4*N + 6 + M})")

    print("\n-- Random valid-action policy run --")
    done = False
    while not done:
        mask = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            print("  No valid actions remaining!")
            break
        a = rng.choice(valid)
        obs, r, done, _, info = env.step(int(a))
        env.render()

    print(f"\nFinal AR          : {info['ar']:.4f}")
    print(f"Services placed   : {info['services_placed']}/{M}")
    print(f"Feasible          : {info['feasible']}")
