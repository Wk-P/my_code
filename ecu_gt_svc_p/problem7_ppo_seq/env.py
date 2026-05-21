"""
P7 Environment — Joint SVC-selection + ECU-assignment (Sequence-to-Sequence ordering).

Unlike P4 which processes SVCs in a fixed sorted order, P7 lets the agent freely
choose WHICH SVC to place at each step AND which ECU to assign it to.
This avoids greedy dead-ends by learning optimal placement ordering.

Action space : Discrete(M × N)
    action = svc_idx * N + ecu_idx
    svc_idx : which unplaced SVC to assign  (0..M-1)
    ecu_idx : which ECU to assign it to     (0..N-1)

Observation (shape 4 + 4N + 3M):
    Global  (4)   : AR, remaining_cap/total_cap, remaining_demand/total_cap, frac_svcs_remaining
    Per-ECU (4N)  : init_cap/max_cap, rem_cap/max_cap, ecu_allowed_frac, svcs_on_ecu/M
    Per-SVC (3M)  : req/max_cap, placed_flag, n_valid_ecus/N
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_ilp.objects import ECU, SVC


class P7Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ecus: list[ECU], services: list[SVC], scenarios=None):
        super().__init__()
        self._scenarios = scenarios
        self.ecus     = ecus
        self.services = services
        self.N = len(ecus)
        self.M = len(services)

        self.action_space = gym.spaces.Discrete(self.M * self.N)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4 + 4 * self.N + 3 * self.M,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.conflict_sets: list[set]
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

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs, _cs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
            # No index remapping needed — services keep original YAML order
            self.conflict_sets = [set(cs) for cs in _cs]
        else:
            self.conflict_sets = self._init_conflict_sets()

        self.N = len(self.ecus)
        self.M = len(self.services)
        self.remaining_vms  = self.initial_vms.copy()
        self.ecu_placements = [set() for _ in range(self.N)]
        self.ecu_allowed    = [set(range(self.M)) for _ in range(self.N)]
        self.placed         = np.zeros(self.M, dtype=bool)
        self.ar             = 0.0
        self._total_ru      = 0.0
        self._step          = 0
        self.capacity_violations = 0
        self.conflict_violations = 0
        self.valid_placed   = 0
        self.episode_has_cap_violation      = False
        self.episode_has_conflict_violation = False
        return self._obs(), {}

    # ── action mask ──────────────────────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.M * self.N, dtype=bool)
        has_valid = False
        for i in range(self.M):
            if self.placed[i]:
                continue
            svc_req = self.services[i].requirement
            for j in range(self.N):
                if (self.remaining_vms[j] >= svc_req) and (not self._has_conflict(j, i)):
                    mask[i * self.N + j] = True
                    has_valid = True
        if has_valid:
            return mask
        # Forced fallback: for each remaining SVC allow the ECU with most remaining capacity.
        best_j = int(np.argmax(self.remaining_vms))
        for i in range(self.M):
            if not self.placed[i]:
                mask[i * self.N + best_j] = True
        return mask

    # ── observation ──────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        max_cap   = float(np.max(self.initial_vms) + 1e-8)
        total_cap = float(np.sum(self.initial_vms) + 1e-8)

        # Global (4)
        remaining_cap_sum = float(np.sum(np.clip(self.remaining_vms, 0, None))) / total_cap
        remaining_dem_sum = float(
            sum(self.services[i].requirement for i in range(self.M) if not self.placed[i])
        ) / total_cap
        frac_remaining = float(np.sum(~self.placed)) / max(self.M, 1)

        # Per-ECU (4N)
        init_cap_frac    = self.initial_vms / max_cap
        rem_cap_frac     = np.clip(self.remaining_vms, -max_cap, max_cap) / max_cap
        ecu_allowed_frac = np.array(
            [len(self.ecu_allowed[j]) / self.M for j in range(self.N)], dtype=np.float32
        )
        svcs_on_ecu_frac = np.array(
            [len(self.ecu_placements[j]) / self.M for j in range(self.N)], dtype=np.float32
        )

        # Per-SVC (3M)
        req_norm    = np.array([s.requirement / max_cap for s in self.services], dtype=np.float32)
        placed_flag = self.placed.astype(np.float32)
        valid_ecus_frac = np.array([
            (sum(
                1 for j in range(self.N)
                if (self.remaining_vms[j] >= self.services[i].requirement)
                and (not self._has_conflict(j, i))
            ) / self.N) if not self.placed[i] else 0.0
            for i in range(self.M)
        ], dtype=np.float32)

        return np.concatenate([
            [self.ar, remaining_cap_sum, remaining_dem_sum, frac_remaining],
            init_cap_frac.astype(np.float32),
            rem_cap_frac.astype(np.float32),
            ecu_allowed_frac,
            svcs_on_ecu_frac,
            req_norm,
            placed_flag,
            valid_ecus_frac,
        ]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        svc_idx = int(action) // self.N
        ecu_idx = int(action) % self.N
        assert 0 <= svc_idx < self.M and not self.placed[svc_idx], (
            f"Invalid action {action}: svc_idx={svc_idx} already_placed={self.placed[svc_idx]}"
        )
        svc = self.services[svc_idx]

        cap_violated      = bool(self.remaining_vms[ecu_idx] < svc.requirement)
        conflict_violated = self._has_conflict(ecu_idx, svc_idx)

        if cap_violated:
            self.capacity_violations += 1
            self.episode_has_cap_violation = True
        if conflict_violated:
            self.conflict_violations += 1
            self.episode_has_conflict_violation = True
        if not (cap_violated or conflict_violated):
            self.valid_placed += 1

        violated          = cap_violated or conflict_violated
        violation_penalty = -2.0 if violated else 0.0
        ru = 0.0 if violated else svc.requirement / (self.initial_vms[ecu_idx] + 1e-8)

        self.remaining_vms[ecu_idx] -= svc.requirement
        self.ecu_placements[ecu_idx].add(svc_idx)
        self._update_ecu_allowed(ecu_idx, svc_idx)
        self.placed[svc_idx] = True
        if ru > 0:
            self._total_ru += ru
        _active = sum(1 for j in range(self.N) if self.ecu_placements[j])
        self.ar = self._total_ru / _active if _active > 0 else 0.0
        self._step += 1

        done       = self._step >= self.M
        total_viol = self.capacity_violations + self.conflict_violations
        terminal_bonus = 0.0
        if done:
            terminal_bonus = self.ar if total_viol == 0 else 0.1 * self.ar

        info = {
            "ar":                             self.ar,
            "step":                           self._step,
            "services_placed":                self._step,
            "valid_placed":                   self.valid_placed,
            "ecus_used":                      _active,
            "capacity_violations":            self.capacity_violations,
            "conflict_violations":            self.conflict_violations,
            "total_violations":               total_viol,
            "violation_rate":                 total_viol / self._step,
            "episode_has_cap_violation":      self.episode_has_cap_violation,
            "episode_has_conflict_violation": self.episode_has_conflict_violation,
        }
        return self._obs(), float(ru + violation_penalty + terminal_bonus), done, False, info

    # ── render ───────────────────────────────────────────────────────────────
    def render(self):
        unplaced = [i for i in range(self.M) if not self.placed[i]]
        print(f"  Step {self._step}/{self.M} | AR={self.ar:.4f} | "
              f"unplaced={len(unplaced)} | "
              f"cap_viol={self.capacity_violations} conf_viol={self.conflict_violations}")


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)
    N, M = 10, 15
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(random.choices(range(50, 200, 5), k=N))]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(random.choices(range(10, 80, 5), k=M))]
    env = P7Env(ecus, services)
    obs, _ = env.reset()
    print(f"Obs shape : {obs.shape}  (expected {4 + 4*N + 3*M})")
    print(f"Action dim: {env.action_space.n}  (expected {M*N})")

    done = False
    while not done:
        mask  = env.action_masks()
        valid = np.where(mask)[0]
        obs, r, done, _, info = env.step(int(valid[0]))
    print(f"AR={info['ar']:.4f}  placed={info['services_placed']}/{M}  "
          f"cap={info['capacity_violations']}  conf={info['conflict_violations']}")
