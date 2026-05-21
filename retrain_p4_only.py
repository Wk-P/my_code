"""
Retrain only P4 (MaskablePPO) for specified groups and seeds,
then patch only the P4_MaskPPO rows in the existing test_results.csv.

Usage:
    python retrain_p4_only.py --group eq gt --seed 1 2 3 4

This script leaves all other models' rows intact.
"""

from __future__ import annotations

import argparse
import csv
import functools
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent

GROUP_META = {
    "eq": {
        "pkg_dir":   ROOT / "ecu_eq_svc_p",
        "yaml_name": "config_ecu_eq_svc.yaml",
    },
    "lt": {
        "pkg_dir":   ROOT / "ecu_lt_svc_p",
        "yaml_name": "config_ecu_lt_svc.yaml",
    },
    "gt": {
        "pkg_dir":   ROOT / "ecu_gt_svc_p",
        "yaml_name": "config_ecu_gt_svc.yaml",
    },
}

TARGET_EPISODES = 2_000_000
_MAX_TIMESTEPS  = 999_999_999
TORCH_THREADS   = 8

_GROUP_MODULES = [
    "problem2_ilp", "problem2_ilp.objects",
    "problem4_ppo_mask", "problem4_ppo_mask.env",
]


def _switch_group(group: str):
    new_pkg = str(GROUP_META[group]["pkg_dir"])
    other_pkgs = {str(GROUP_META[g]["pkg_dir"]) for g in GROUP_META if g != group}
    sys.path[:] = [p for p in sys.path if p not in other_pkgs]
    if new_pkg not in sys.path:
        sys.path.insert(0, new_pkg)
    for mod in list(sys.modules.keys()):
        if any(mod == m or mod.startswith(m + ".") for m in _GROUP_MODULES):
            del sys.modules[mod]


def _load_scenarios(group: str) -> list:
    meta = GROUP_META[group]
    yaml_path = meta["pkg_dir"] / "problem2_ilp" / "config" / meta["yaml_name"]
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return [
        (
            [ecu["capacity"]    for ecu in sc["ECUs"]],
            [svc["requirement"] for svc in sc["SVCs"]],
            sc.get("conflict_sets", []),
        )
        for sc in cfg["scenarios"]
    ]


def _make_split(scenarios: list, seed: int, train_ratio: float = 0.8):
    rng  = random.Random(seed)
    idxs = list(range(len(scenarios)))
    rng.shuffle(idxs)
    n_train = int(train_ratio * len(idxs))
    return [scenarios[i] for i in idxs[:n_train]], [scenarios[i] for i in idxs[n_train:]]


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── episode-count stopping callback ──────────────────────────────────────────

class EpisodeTrackingCallback:
    def __init__(self, target_episodes: int):
        from stable_baselines3.common.callbacks import BaseCallback
        class _Cb(BaseCallback):
            def __init__(self, target):
                super().__init__()
                self.target = target
                self.episode_count   = 0
                self.episode_rewards: list[float] = []
            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        self.episode_count += 1
                        self.episode_rewards.append(float(info["episode"]["r"]))
                return self.episode_count < self.target
        self._cb = _Cb(target_episodes)

    def __getattr__(self, name):
        return getattr(self._cb, name)

    def unwrap(self):
        return self._cb


def _make_p4_env(P4Env, ECU, SVC, seed_i, train_scenarios):
    from stable_baselines3.common.monitor import Monitor
    from sb3_contrib.common.wrappers import ActionMasker
    random.seed(seed_i)
    caps, reqs, _ = train_scenarios[0]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env  = P4Env(ecus, svcs, scenarios=train_scenarios)
    env  = ActionMasker(env, lambda e: e.action_masks())
    return Monitor(env)


def train_p4(P4Env, ECU, SVC, train_scenarios, seed, device, target_episodes):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    class _EpCb(BaseCallback):
        def __init__(self, target):
            super().__init__()
            self.target = target
            self.episode_count = 0
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episode_count += 1
            return self.episode_count < self.target

    torch.set_num_threads(TORCH_THREADS)
    ep_cb = _EpCb(target_episodes)
    vec   = DummyVecEnv([
        functools.partial(_make_p4_env, P4Env, ECU, SVC, seed + i, train_scenarios)
        for i in range(40)
    ])
    model = MaskablePPO(
        policy="MlpPolicy", env=vec, learning_rate=3e-4, n_steps=256,
        batch_size=128, n_epochs=10, gamma=0.999, gae_lambda=0.95,
        clip_range=0.2, policy_kwargs=dict(net_arch=[256, 256]),
        device=device, verbose=0, seed=seed,
    )
    t0 = time.time()
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=ep_cb)
    vec.close()
    print(f"  [P4] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model


def evaluate_p4(model, test_scenarios, P4Env, ECU, SVC):
    results = []
    for scenario in test_scenarios:
        caps, reqs, cs = scenario
        ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
        svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
        env  = P4Env(ecus, svcs, scenarios=[scenario])
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            mask = env.action_masks()
            if not np.any(mask):
                break
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        M = len(svcs)
        placed  = int(info.get("services_placed", 0))
        has_cap  = bool(info.get("episode_has_cap_violation", False))
        has_conf = bool(info.get("episode_has_conflict_violation", False))
        success  = int(placed == M and not has_cap and not has_conf)
        results.append({
            "ar":                             round(float(info.get("ar", 0.0)), 6),
            "success":                        success,
            "services_placed":                placed,
            "M":                              M,
            "episode_has_cap_violation":      int(has_cap),
            "episode_has_conflict_violation": int(has_conf),
        })
    return results


def patch_csv(csv_path: Path, new_results: list[dict]):
    """Replace only P4_MaskPPO rows; keep all other rows intact."""
    with open(csv_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    fieldnames = ["model", "scenario_idx", "ar", "success",
                  "services_placed", "M",
                  "episode_has_cap_violation", "episode_has_conflict_violation"]

    kept  = [r for r in all_rows if r["model"] != "P4_MaskPPO"]
    new   = [
        {
            "model":                          "P4_MaskPPO",
            "scenario_idx":                   idx,
            "ar":                             r["ar"],
            "success":                        r["success"],
            "services_placed":                r["services_placed"],
            "M":                              r["M"],
            "episode_has_cap_violation":      r["episode_has_cap_violation"],
            "episode_has_conflict_violation": r["episode_has_conflict_violation"],
        }
        for idx, r in enumerate(new_results)
    ]

    # Preserve original model order
    MODEL_ORDER = ["P3_PPO", "P4_MaskPPO", "P5_LagPPO", "P6_RepairPPO", "DQN", "DDQN"]
    combined = kept + new
    combined.sort(key=lambda r: (MODEL_ORDER.index(r["model"])
                                  if r["model"] in MODEL_ORDER else 99,
                                  int(r["scenario_idx"])))

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(combined)
    print(f"  Patched → {csv_path}")


def run_one(group: str, seed: int, target_episodes: int):
    print(f"\n{'='*60}")
    print(f"  Retraining P4  group={group}  seed={seed}")
    print(f"{'='*60}")

    _switch_group(group)
    from problem2_ilp.objects  import ECU, SVC
    from problem4_ppo_mask.env import P4Env

    scenarios = _load_scenarios(group)
    train_sc, test_sc = _make_split(scenarios, seed)
    device = _resolve_device()
    print(f"  Split: {len(train_sc)} train / {len(test_sc)} test  |  device={device}")

    model = train_p4(P4Env, ECU, SVC, train_sc, seed, device, target_episodes)

    print("  Evaluating …")
    results = evaluate_p4(model, test_sc, P4Env, ECU, SVC)
    n_conf = sum(r["episode_has_conflict_violation"] for r in results)
    n_cap  = sum(r["episode_has_cap_violation"]      for r in results)
    n_ok   = sum(r["success"]                         for r in results)
    ar_mean = np.mean([r["ar"] for r in results])
    print(f"  AR={ar_mean:.4f}  success={n_ok}/{len(results)}  "
          f"cap_viol={n_cap}  conf_viol={n_conf}")

    outdir  = ROOT / "results" / group / f"seed_{seed}"
    csv_path = outdir / "test_results.csv"
    if csv_path.exists():
        patch_csv(csv_path, results)
    else:
        print(f"  WARNING: {csv_path} not found — skipping patch")

    print(f"  Done  group={group}  seed={seed}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group",    nargs="+", default=["eq", "gt"],
                        choices=["eq", "lt", "gt"])
    parser.add_argument("--seed",     nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--episodes", type=int,  default=TARGET_EPISODES)
    args = parser.parse_args()

    for group in args.group:
        for seed in args.seed:
            run_one(group, seed, args.episodes)

    print("All retraining complete.")


if __name__ == "__main__":
    main()
