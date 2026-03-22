"""
DQN Environment — RL WITHOUT action masking.

Constraint handling and reward shaping:
    - If an invalid ECU is chosen (capacity insufficient OR already assigned),
        the episode terminates immediately with a penalty proportional to the
        number of services still left unplaced.
    - Each valid assignment receives its exact utilisation contribution
        n_i / e_j. This aligns the return with the ILP objective instead of only
        rewarding local AR direction changes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_ilp.objects import ECU, SVC


class DQNEnv(gym.Env):
    """
    Service Deployment Environment WITHOUT action masking.

        Observation (shape: 3N+2):
      [0]   current service demand, normalised
      [1]   current cumulative AR
            [2:2+N]      remaining capacity fraction per ECU
            [2+N:2+2N]   ECU occupied flag per ECU (1 = already assigned)
            [2+2N:2+3N]  valid-action flag for current service (1 = feasible now)

        Reward:
            -k     constraint violated (capacity exceeded OR duplicate ECU), where
                         k is the number of services still left unplaced
            +ru    valid assignment contribution, ru = requirement / ecu_capacity
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
            low=0.0, high=1.0, shape=(3 * self.N + 2,), dtype=np.float32,
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
        self.remaining_vms = self.initial_vms.copy()
        self.ecu_assigned  = np.zeros(self.N, dtype=bool)
        self.ar    = 0.0
        self._step = 0
        return self._obs(), {}

    # ── observation ──────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        if self._step >= self.M:
            service_demand_norm = np.float32(0.0)
        else:
            svc = self.services[self._step]
            service_demand_norm = np.float32(svc.requirement / (np.max(self.initial_vms) + 1e-8))
        remaining_pct = self.remaining_vms / (self.initial_vms + np.float32(1e-8))
        assigned_flag = self.ecu_assigned.astype(np.float32)
        if self._step >= self.M:
            valid_flag = np.zeros(self.N, dtype=np.float32)
        else:
            valid_flag = ((~self.ecu_assigned) & (self.remaining_vms >= svc.requirement)).astype(np.float32)
        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            remaining_pct,
            assigned_flag,
            valid_flag,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        svc = self.services[self._step]

        # Constraint check: capacity violation OR duplicate ECU → fail immediately
        if self.remaining_vms[action] < svc.requirement or self.ecu_assigned[action]:
            remaining_services = self.M - self._step
            return self._obs(), -float(remaining_services), True, False, {
                "ar":              self.ar,
                "services_placed": self._step,
                "violated":        True,
            }

        # ── Valid assignment ──────────────────────────────────────────────────
        ru = svc.requirement / (self.initial_vms[action] + 1e-8)
        self.remaining_vms[action] -= svc.requirement
        self.ecu_assigned[action]   = True

        prev_ar = self.ar
        self.ar = (self.ar * self._step + ru) / (self._step + 1)
        self._step += 1

        done   = self._step >= self.M


        reward = float(ru)

        return self._obs(), reward, done, False, {
            "ar":              self.ar,
            "step":            self._step,
            "services_placed": self._step,
            "violated":        False,
        }

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs "
                  f"| AR={self.ar:.4f}")
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

    env = DQNEnv(ecus, services)
    obs, _ = env.reset()
    print(f"\nObs shape : {obs.shape}  (expected {3 * N + 2})")

    print("\n-- Valid greedy policy run --")
    done = False
    while not done:
        svc = env.services[env._step]
        valid = [j for j in range(N)
                 if not env.ecu_assigned[j] and env.remaining_vms[j] >= svc.requirement]
        if not valid:
            print("  No valid ECU found!")
            break
        a = valid[0]
        obs, r, done, _, info = env.step(a)
        env.render()

    print(f"\nFinal AR          : {info['ar']:.4f}")
    print(f"Services placed   : {info['services_placed']}/{M}")
    print(f"Violated          : {info.get('violated', False)}")
