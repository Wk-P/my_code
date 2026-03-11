"""
P3 Environment — Service Deployment on 200 randomised scenarios.

Same constraint logic as env/env.py:
  1. Capacity violation  → reward = -1, episode terminates immediately.
  2. Single-ECU rule     → same: reward = -1, episode terminates.
  3. Reward              → sparse final AR on successful completion.
  4. Scenarios           → reset() randomly picks one of 200 scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_single.objects import ECU, SVC


# ─────────────────────────────────────────────────────────────────────────────
#  P3 Environment
# ─────────────────────────────────────────────────────────────────────────────

class P3Env(gym.Env):
    """
    Each episode assigns M services to N ECUs, one service per step.

    Constraint violations → reward = -1, episode terminates immediately:
      • capacity_violations      : remaining_vms[j] < service.requirement
      • single_service_violations: ECU j already had a service assigned

    Reward:
      • 0.0  for every intermediate step
      • -1.0 on any constraint violation (episode ends)
      • final AR at the last step (step M-1) on clean completion

    Observation  (shape: N+2):
      [0]   current service demand, normalised by max initial capacity  ∈ [0,1]
      [1]   current cumulative AR                                        ∈ [0,1]
      [2:]  remaining capacity fraction per ECU
    """

    metadata = {"render_modes": []}

    def __init__(self, ecus: list[ECU], services: list[SVC], scenarios=None):
        """
        scenarios: 可选，list of (caps_list, reqs_list)。
        若提供，每次 reset() 将从中随机选取一个 scenario。
        """
        super().__init__()
        self._scenarios = scenarios   # None → 固定单 scenario
        self.ecus     = ecus
        self.services = services
        self.N = len(ecus)
        self.M = len(services)

        # action: choose which ECU to assign the current service to
        self.action_space = gym.spaces.Discrete(self.N)

        self.observation_space = gym.spaces.Box(
            low  =  0.0,
            high =  1.0,
            shape= (self.N + 2,),
            dtype= np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)

        # mutable state (initialised in reset)
        self.remaining_vms: np.ndarray
        self.ecu_assigned:  np.ndarray   # bool: has this ECU been used?
        self.ar:            float
        self._step:         int

        # call reset to initialise
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
            service_demand_norm = 0.0
        else:
            svc = self.services[self._step]
            service_demand_norm = svc.requirement / (np.max(self.initial_vms) + 1e-8)

        remaining_pct = self.remaining_vms / (self.initial_vms + np.float32(1e-8))

        return np.concatenate([
            np.array([service_demand_norm], dtype=np.float32),
            np.array([self.ar], dtype=np.float32),
            remaining_pct,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert 0 <= action < self.N, f"Invalid action {action}"
        svc = self.services[self._step]

        # ── Constraint check: violate → penalty and terminate ────────────────
        if self.remaining_vms[action] < svc.requirement:
            self.capacity_violations += 1
            self._step += 1
            return self._obs(), -1.0, True, False, {
                "ar": self.ar, "step": self._step,
                "capacity_violations": self.capacity_violations,
                "single_service_violations": self.single_service_violations,
                "violation_rate": (self.capacity_violations + self.single_service_violations) / self._step,
                "violation": "capacity",
            }
        if self.ecu_assigned[action]:
            self.single_service_violations += 1
            self._step += 1
            return self._obs(), -1.0, True, False, {
                "ar": self.ar, "step": self._step,
                "capacity_violations": self.capacity_violations,
                "single_service_violations": self.single_service_violations,
                "violation_rate": (self.capacity_violations + self.single_service_violations) / self._step,
                "violation": "duplicate",
            }

        # ── Valid assignment ──────────────────────────────────────────────────
        ru = svc.requirement / (self.initial_vms[action] + 1e-8)
        self.remaining_vms[action] -= svc.requirement
        self.ecu_assigned[action]   = True

        # Incremental AR update
        self.ar = (self.ar * self._step + ru) / (self._step + 1)
        self._step += 1

        done   = self._step >= self.M
        reward = float(self.ar) if done else 0.0

        info = {
            "ar":   self.ar,
            "step": self._step,
            "capacity_violations":       self.capacity_violations,
            "single_service_violations": self.single_service_violations,
            "violation_rate":            0.0,
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
    print(f"Violation rate        : {info['violation_rate']:.2%}")
