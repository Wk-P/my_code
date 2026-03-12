"""
P4 Environment — RL WITH constraint enforcement via Action Masking.

Key difference from P3:
  - action_masks() returns a boolean array: True = valid ECU, False = invalid.
  - Invalid = capacity insufficient OR ECU already assigned.
  - Agent can ONLY choose from valid actions → 0 constraint violations guaranteed.
  - If no valid action exists (infeasible), episode terminates with reward = current AR.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_single.objects import ECU, SVC


class P4Env(gym.Env):
    """
    Service Deployment Environment WITH action masking.

    action_masks() is called by MaskablePPO before each step to know
    which actions are legal. The agent's policy logits for masked actions
    are set to -inf, so they are never selected.

    Observation (shape: N+2):
      [0]   current service demand, normalised
      [1]   current cumulative AR
      [2:]  remaining capacity fraction per ECU

    Reward:
      0.0   for intermediate steps
      AR    at the final step (or when no valid action remains)
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
            low=0.0, high=1.0, shape=(self.N + 2,), dtype=np.float32,
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
        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            remaining_pct,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert 0 <= action < self.N, f"Invalid action {action}"
        svc = self.services[self._step]

        # With action masking this should never happen, but defend just in case
        if self.remaining_vms[action] < svc.requirement or self.ecu_assigned[action]:
            # Infeasible action somehow got through → terminate with current AR
            self._step += 1
            done = True
            return self._obs(), float(self.ar), done, False, {
                "ar": self.ar, "step": self._step,
                "feasible": False,
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

        reward = float(self.ar) if done else 0.0

        info = {
            "ar":       self.ar,
            "step":     self._step,
            "feasible": True,
            "services_placed": self._step,
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
    print(f"\nObs shape : {obs.shape}  (expected {N+2})")

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
