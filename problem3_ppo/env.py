"""
P3 Environment — RL WITHOUT constraint enforcement.

Design intent (matches docs.md):
  • Constraint violations are RECORDED but NOT penalized.
  • Episode NEVER terminates early — always runs M steps.
  • Infeasible assignments still happen (remaining_vms can go negative).
  • Reward: sparse final AR at step M-1, 0.0 for intermediate steps.
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
#  P3 Environment — NO constraint enforcement
# ─────────────────────────────────────────────────────────────────────────────

class P3Env(gym.Env):
    """
    Each episode assigns M services to N ECUs, one service per step.

    Constraints are NOT enforced:
      • capacity violation   → only recorded, assignment still happens
      • duplicate ECU usage  → only recorded, assignment still happens
      • remaining_vms can go negative (overloaded)

    Reward:
      • +1.0 if AR increased vs. previous step
      •  0.0 otherwise
      (no terminal lump-sum; signal is dense and comparison-based)

    Observation  (shape: N+2):
      [0]   current service demand, normalised by max initial capacity
      [1]   current cumulative AR
      [2:]  remaining capacity fraction per ECU (can be negative if overloaded)
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

        # obs can go slightly below 0 (overloaded ECU) so low=-1
        self.observation_space = gym.spaces.Box(
            low  = -1.0,
            high =  1.0,
            shape= (self.N + 2,),
            dtype= np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms: np.ndarray
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
        self.ecu_assigned   = np.zeros(self.N, dtype=bool)
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

        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            remaining_pct,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert 0 <= action < self.N, f"Invalid action {action}"
        svc = self.services[self._step]

        # ── Constraint check: ONLY record, do NOT penalize or terminate ──────
        if self.remaining_vms[action] < svc.requirement:
            self.capacity_violations += 1
        if self.ecu_assigned[action]:
            self.single_service_violations += 1

        # ── Assignment always happens regardless of violations ────────────────
        ru = svc.requirement / (self.initial_vms[action] + 1e-8)
        self.remaining_vms[action] -= svc.requirement      # can go negative
        self.ecu_assigned[action]   = True

        # Incremental AR update
        prev_ar = self.ar
        self.ar = (self.ar * self._step + ru) / (self._step + 1)
        self._step += 1

        done   = self._step >= self.M
        # Reward: +1 if AR improved, -1 if AR dropped, 0 if unchanged.
        if self.ar > prev_ar:
            reward = 1.0
        elif self.ar < prev_ar:
            reward = -1.0
        else:
            reward = 0.0

        total_viol = self.capacity_violations + self.single_service_violations
        info = {
            "ar":   self.ar,
            "step": self._step,
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

    env = P3Env(ecus, services)
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
