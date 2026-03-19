"""
P5 Environment — Lagrangian Constraint Relaxation.

Constraints are SOFT:
  - Violations are NOT blocking; episode always runs M steps.
  - Each violated step incurs per-step penalty: lambda_val * c_t.
  - lambda_val (Lagrangian multiplier) is updated externally by the training
    callback via dual ascent: λ ← clip(λ + lr*(avg_viol - target), 0, λ_max).

Violation at step t (c_t):
  c_t = 1.0  if capacity insufficient OR ECU already assigned, else 0.0

Reward per step:
  r_t = improvement_signal  -  lambda_val * c_t
  where improvement_signal = +1.0 if AR increased vs. previous step, else 0.0

The improvement signal replaces the old ru/M dense reward.
remaining_vms can go negative (overloaded ECU).
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
    Observation (shape: N+2):
      [0]   current service demand, normalised by max initial capacity
      [1]   current cumulative AR
      [2:]  remaining capacity fraction per ECU (can be < 0 if overloaded)

    Reward per step:
      r_t = improvement_signal  -  lambda_val * c_t
      improvement_signal = +1.0 if AR increased vs. previous step, else 0.0
      c_t = 1.0 if capacity insufficient OR ECU already used, else 0.0

    Lagrangian multiplier λ is updated externally by the training callback via dual ascent:
    λ ← clip(λ + lr*(avg_viol - target), 0, λ_max)
    """

    metadata = {"render_modes": []}

    def __init__(self, ecus: list[ECU], services: list[SVC],
                 scenarios=None, lambda_init: float = 0.0):
        super().__init__()
        self._scenarios = scenarios
        self.ecus       = ecus
        self.services   = services
        self.N          = len(ecus)
        self.M          = len(services)
        self.lambda_val = float(lambda_init)

        self.action_space = gym.spaces.Discrete(self.N)
        # remaining_pct can be negative when overloaded
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=1.0, shape=(self.N + 2,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms:      np.ndarray
        self.ecu_assigned:       np.ndarray
        self.ar:                 float
        self._step:              int
        self.episode_violations: int
        self.reset()

    # ── λ setter — called by training callback after each dual-ascent step ────
    def set_lambda(self, val: float):
        self.lambda_val = float(val)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
        self.remaining_vms      = self.initial_vms.copy()
        self.ecu_assigned       = np.zeros(self.N, dtype=bool)
        self.ar                 = 0.0
        self._step              = 0
        self.episode_violations = 0
        return self._obs(), {}

    # ── observation ───────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        if self._step >= self.M:
            service_demand_norm = np.float32(0.0)
        else:
            svc = self.services[self._step]
            service_demand_norm = np.float32(
                svc.requirement / (np.max(self.initial_vms) + 1e-8)
            )
        remaining_pct = self.remaining_vms / (self.initial_vms + np.float32(1e-8))
        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            remaining_pct,
        ]).astype(np.float32)

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        svc = self.services[self._step]

        # ── Violation check (soft — assignment always proceeds) ───────────────
        cap_violated = bool(self.remaining_vms[action] < svc.requirement)
        dup_violated = bool(self.ecu_assigned[action])
        violated     = cap_violated or dup_violated
        c_t          = 1.0 if violated else 0.0

        # ── Perform assignment regardless of violations ───────────────────────
        ru = float(svc.requirement / (self.initial_vms[action] + 1e-8))
        self.remaining_vms[action] -= svc.requirement   # may go negative
        self.ecu_assigned[action]   = True

        prev_ar = self.ar
        self.ar = (self.ar * self._step + ru) / (self._step + 1)
        self._step += 1
        if violated:
            self.episode_violations += 1

        done = self._step >= self.M

        # ── Lagrangian reward: AR direction signal − λ·violation ─────────────
        delta_ar = self.ar - prev_ar
        reward = delta_ar - self.lambda_val * c_t

        return self._obs(), reward, done, False, {
            "ar":              self.ar,
            "violated":        violated,
            "violations_ep":   self.episode_violations,
            "viol_rate_ep":    self.episode_violations / self._step,
            "services_placed": self._step,
            "lambda":          self.lambda_val,
        }

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs"
                  f" | AR={self.ar:.4f} | λ={self.lambda_val:.3f}"
                  f" | violations={self.episode_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f}"
                  f" | violations={self.episode_violations}/{self.M}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
