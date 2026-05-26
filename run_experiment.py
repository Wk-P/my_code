"""
Unified experiment runner — one seed, one group (eq / lt / gt), all 6 models.

Supports all three scenario groups by dynamically resolving the correct package
directory and YAML config for each group.

Usage:
    python run_experiment.py --seed 3 4 5 --group all --steps 5000000
    python run_experiment.py --seed 1 --group eq
    python run_experiment.py --seed 1 2 --group eq     # multiple seeds sequentially
    python run_experiment.py --seed 1 --group lt
    python run_experiment.py --seed 1 --group gt

Outputs (per seed):
    results/<group>/seed_<N>/
        training_curve.png     — all models' reward vs episode on one figure
        test_results.png       — AR / success-failure / violation subplots
        training_rewards.csv   — per-episode rewards for each model
        test_results.csv       — per-scenario test metrics for each model
        test_results_agg.csv   — aggregated metrics per model
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent

# ── group → package directory + YAML config ───────────────────────────────────
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

# ── global training target ────────────────────────────────────────────────────
TARGET_STEPS    = 5_000_000   # converted to episodes per group via steps // M
_MAX_TIMESTEPS  = 999_999_999
TORCH_THREADS   = 16
PPO_N_STEPS     = 512
PPO_BATCH_SIZE  = 256


# ══════════════════════════════════════════════════════════════════════════════
#  dynamic imports — must be called after sys.path is set for the group
# ══════════════════════════════════════════════════════════════════════════════

def _import_group_modules():
    """Import env classes from the currently-active group package."""
    from problem2_ilp.objects           import ECU, SVC
    from problem3_ppo.env               import P3Env
    from problem4_ppo_mask.env          import P4Env
    from problem5_ppo_lagrangian.env    import LagrangeEnv
    from problem6_ppo_opt.env           import P6Env
    from problem_dqn.env                import DQNEnv
    from problem_ddqn.env               import DDQNEnv
    from problem_ddqn.run_all           import DDQN
    return ECU, SVC, P3Env, P4Env, LagrangeEnv, P6Env, DQNEnv, DDQNEnv, DDQN


def _import_experiment_modules():
    """Import shared experiment utilities from the root-level experiment/ package."""
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from experiment.callbacks  import EpisodeTrackingCallback, LagrangianUpdateCallback
    from experiment.eval_utils import evaluate_model, aggregate_eval
    from experiment.plots      import plot_training_curves, plot_test_results
    return (EpisodeTrackingCallback, LagrangianUpdateCallback,
            evaluate_model, aggregate_eval,
            plot_training_curves, plot_test_results)


# ══════════════════════════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_split(scenarios: list, seed: int, train_ratio: float = 0.8):
    rng  = random.Random(seed)
    idxs = list(range(len(scenarios)))
    rng.shuffle(idxs)
    n_train = int(train_ratio * len(idxs))
    return [scenarios[i] for i in idxs[:n_train]], [scenarios[i] for i in idxs[n_train:]]


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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


# ══════════════════════════════════════════════════════════════════════════════
#  environment factories
# ══════════════════════════════════════════════════════════════════════════════

def _make_env(env_cls, ECU, SVC, seed_i, train_scenarios, extra_kw=None):
    from stable_baselines3.common.monitor import Monitor
    random.seed(seed_i)
    caps, reqs, _ = train_scenarios[0]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env  = env_cls(ecus, svcs, scenarios=train_scenarios, **(extra_kw or {}))
    return Monitor(env)


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


# ══════════════════════════════════════════════════════════════════════════════
#  training functions
# ══════════════════════════════════════════════════════════════════════════════

def _train_ppo(vec_env, ep_cb, extra_cbs, seed, device, ppo_kwargs):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    torch.set_num_threads(TORCH_THREADS)
    cbs = [ep_cb] + (extra_cbs or [])
    model = PPO(policy="MlpPolicy", env=vec_env, device=device,
                verbose=0, seed=seed, **ppo_kwargs)
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=CallbackList(cbs))
    vec_env.close()
    return model


def _make_vecenv(env_fn_list):
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv(env_fn_list)


def train_p3(ECU, SVC, P3Env, EpisodeTrackingCallback,
             train_scenarios, seed, device, target_episodes):
    M = len(train_scenarios[0][1])
    print(f"  [P3 PPO] training … M={M} n_steps=512 batch=256")
    t0 = time.time()
    ep_cb = EpisodeTrackingCallback(target_episodes)
    vec   = _make_vecenv([
        functools.partial(_make_env, P3Env, ECU, SVC, seed + i, train_scenarios)
        for i in range(40)
    ])
    model = _train_ppo(vec, ep_cb, [], seed, device, dict(
        learning_rate=3e-4, n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE, n_epochs=10,
        gamma=0.999, gae_lambda=0.95, clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
    ))
    print(f"  [P3 PPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def train_p4(ECU, SVC, P4Env, EpisodeTrackingCallback,
             train_scenarios, seed, device, target_episodes):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    M = len(train_scenarios[0][1])
    print(f"  [P4 MaskPPO] training … M={M} n_steps=512 batch=256")
    t0 = time.time()
    torch.set_num_threads(TORCH_THREADS)
    ep_cb = EpisodeTrackingCallback(target_episodes)
    vec   = DummyVecEnv([
        functools.partial(_make_p4_env, P4Env, ECU, SVC, seed + i, train_scenarios)
        for i in range(40)
    ])
    model = MaskablePPO(
        policy="MlpPolicy", env=vec, learning_rate=3e-4, n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
        device=device, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=ep_cb)
    vec.close()
    print(f"  [P4 MaskPPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def train_p5(ECU, SVC, LagrangeEnv, EpisodeTrackingCallback, LagrangianUpdateCallback,
             train_scenarios, seed, device, target_episodes):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    M = len(train_scenarios[0][1])
    print(f"  [P5 LagPPO] training … M={M} n_steps=512 batch=256")
    t0 = time.time()
    torch.set_num_threads(TORCH_THREADS)
    LAMBDA_INIT, LAMBDA_MAX, LAMBDA_LR, LAMBDA_TARGET, LAMBDA_WIN = 0.5, 5.0, 0.01, 0.0, 200
    ep_cb  = EpisodeTrackingCallback(target_episodes)
    lam_cb = LagrangianUpdateCallback(LAMBDA_INIT, LAMBDA_LR, LAMBDA_TARGET, LAMBDA_MAX, LAMBDA_WIN)
    vec = _make_vecenv([
        functools.partial(_make_env, LagrangeEnv, ECU, SVC, seed + i, train_scenarios,
                          {"lambda_init": LAMBDA_INIT, "lambda_max": LAMBDA_MAX})
        for i in range(40)
    ])
    model = PPO(
        policy="MlpPolicy", env=vec, learning_rate=3e-4, n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
        device=device, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=CallbackList([ep_cb, lam_cb]))
    vec.close()
    print(f"  [P5 LagPPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def train_p6(ECU, SVC, P6Env, EpisodeTrackingCallback,
             train_scenarios, seed, device, target_episodes):
    M = len(train_scenarios[0][1])
    print(f"  [P6 RepairPPO] training … M={M} n_steps=512 batch=256")
    t0 = time.time()
    ep_cb = EpisodeTrackingCallback(target_episodes)
    vec   = _make_vecenv([
        functools.partial(_make_env, P6Env, ECU, SVC, seed + i, train_scenarios)
        for i in range(40)
    ])
    model = _train_ppo(vec, ep_cb, [], seed, device, dict(
        learning_rate=3e-4, n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
    ))
    print(f"  [P6 RepairPPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def _train_dqn_variant(model_cls, env_cls, name,
                       ECU, SVC, EpisodeTrackingCallback,
                       train_scenarios, seed, device, target_episodes):
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(TORCH_THREADS)
    ep_cb = EpisodeTrackingCallback(target_episodes)
    vec   = DummyVecEnv([
        functools.partial(_make_env, env_cls, ECU, SVC, seed + i, train_scenarios)
        for i in range(12)
    ])
    model = model_cls(
        policy="MlpPolicy", env=vec, learning_rate=1e-3,
        buffer_size=100_000, learning_starts=64, batch_size=64,
        tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1,
        target_update_interval=500, exploration_fraction=0.1,
        exploration_final_eps=0.0,
        policy_kwargs=dict(net_arch=[128, 128]),
        device=device, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=ep_cb)
    vec.close()
    return model, ep_cb


def train_dqn(ECU, SVC, DQNEnv, EpisodeTrackingCallback,
              train_scenarios, seed, device, target_episodes):
    from stable_baselines3 import DQN
    print("  [DQN] training …")
    t0 = time.time()
    model, ep_cb = _train_dqn_variant(
        DQN, DQNEnv, "DQN", ECU, SVC, EpisodeTrackingCallback,
        train_scenarios, seed, device, target_episodes,
    )
    print(f"  [DQN] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def train_ddqn(ECU, SVC, DDQNEnv, DDQN, EpisodeTrackingCallback,
               train_scenarios, seed, device, target_episodes):
    print("  [DDQN] training …")
    t0 = time.time()
    model, ep_cb = _train_dqn_variant(
        DDQN, DDQNEnv, "DDQN", ECU, SVC, EpisodeTrackingCallback,
        train_scenarios, seed, device, target_episodes,
    )
    print(f"  [DDQN] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


# ══════════════════════════════════════════════════════════════════════════════
#  CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_training_csv(training_data: dict, outdir: Path):
    path = outdir / "training_rewards.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "episode", "reward"])
        for name, data in training_data.items():
            for ep, rew in zip(data["episode_nums"], data["episode_rewards"]):
                w.writerow([name, ep, round(rew, 6)])
    print(f"  Saved → {path}")


def _save_test_csv(all_eval: dict, outdir: Path):
    path = outdir / "test_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "scenario_idx", "ar", "success",
                    "services_placed", "M",
                    "episode_has_cap_violation", "episode_has_conflict_violation"])
        for name, results in all_eval.items():
            for idx, r in enumerate(results):
                w.writerow([name, idx, round(r["ar"], 6), int(r["success"]),
                             r["services_placed"], r["M"],
                             int(r["episode_has_cap_violation"]),
                             int(r["episode_has_conflict_violation"])])
    print(f"  Saved → {path}")


def _save_agg_csv(agg: dict, outdir: Path):
    path = outdir / "test_results_agg.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ar_mean", "ar_std", "success_rate", "failure_rate",
                    "cap_viol_rate", "conf_viol_rate", "placed_mean", "M", "n_scenarios"])
        for name, d in agg.items():
            w.writerow([name, round(d["ar_mean"], 6), round(d["ar_std"], 6),
                        round(d["success_rate"], 6), round(d["failure_rate"], 6),
                        round(d["cap_viol_rate"], 6), round(d["conf_viol_rate"], 6),
                        round(d["placed_mean"], 4), d["M"], d["n_scenarios"]])
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  main experiment runner (one seed)
# ══════════════════════════════════════════════════════════════════════════════

MODELS_ORDERED = ["P3_PPO", "P4_MaskPPO", "P5_LagPPO", "P6_RepairPPO", "DQN", "DDQN"]


class _Tee:
    """Write to multiple file-like objects simultaneously."""
    def __init__(self, *files):
        self._files = files
    def write(self, data):
        for f in self._files:
            f.write(data)
    def flush(self):
        for f in self._files:
            f.flush()
    def fileno(self):
        return self._files[0].fileno()


def _load_ilp(group: str) -> float | None:
    """Read cached ILP average utilisation for a group (returns None if not found)."""
    GROUP_ENV = {"lt": "ecu_lt_svc_p", "eq": "ecu_eq_svc_p", "gt": "ecu_gt_svc_p"}
    ILP_PROBE = ["problem4_ppo_mask", "problem3_ppo", "problem_dqn", "problem6_ppo_opt"]
    env_dir = ROOT / GROUP_ENV.get(group, "")
    for prob in ILP_PROBE:
        cache = env_dir / prob / "results" / "ilp_cache.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                ars  = [r["avg_utilization"] for r in data.get("results", [])
                        if "avg_utilization" in r]
                if ars:
                    return round(float(np.mean(ars)), 4)
            except Exception:
                pass
    return None


def _compute_and_save_ilp(group: str, scenarios: list, exp_root: Path) -> float | None:
    """Compute ILP optimal AR for a group and persist to reports; skip if already done."""
    ilp_json = exp_root / group.upper() / "results" / "ilp.json"
    if ilp_json.exists():
        try:
            return float(json.loads(ilp_json.read_text())["ilp_ar"])
        except Exception:
            pass

    val = _load_ilp(group)
    if val is None:
        _switch_group(group)
        if "run_utils" in sys.modules:
            del sys.modules["run_utils"]
        try:
            from run_utils import solve_ilp_all_scenarios
            meta      = GROUP_META[group]
            yaml_path = meta["pkg_dir"] / "problem2_ilp" / "config" / meta["yaml_name"]
            outdir    = meta["pkg_dir"] / "results"
            print(f"  [ILP] computing for {group.upper()} ({len(scenarios)} scenarios) …")
            mean_ar, _ = solve_ilp_all_scenarios(yaml_path, scenarios, outdir)
            val = round(float(mean_ar), 4)
            print(f"  [ILP] {group.upper()}: mean AR = {val}")
        except Exception as e:
            print(f"  [ILP] warning: could not compute ILP for {group}: {e}")
            return None

    ilp_json.parent.mkdir(parents=True, exist_ok=True)
    ilp_json.write_text(json.dumps({"ilp_ar": val}))
    return val


def _generate_summary(exp_root: Path, groups: list[str]):
    """Generate per-group cross-seed summary figures (fig1–4) into {GROUP}/results/figures/."""
    print(f"\n{'='*60}\n  Generating cross-seed summary figures …\n{'='*60}")

    # Reset sys.path: remove all group pkgs, ensure ROOT is at front
    for g in GROUP_META:
        pkg = str(GROUP_META[g]["pkg_dir"])
        while pkg in sys.path:
            sys.path.remove(pkg)
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    for mod in list(sys.modules):
        if mod.startswith("experiment"):
            del sys.modules[mod]

    from experiment.plot_summary import load_groups_agg, plot_experiment_summary

    for g in groups:
        seeds_root = exp_root / g.upper() / "seeds"
        if not seeds_root.exists():
            continue
        dirs = sorted(
            [d for d in seeds_root.iterdir() if d.is_dir()],
            key=lambda p: int(p.name) if p.name.isdigit() else 0,
        )
        if not dirs:
            continue

        group_agg = load_groups_agg({g: dirs})
        if not group_agg:
            continue

        ilp_val  = None
        ilp_json = exp_root / g.upper() / "results" / "ilp.json"
        if ilp_json.exists():
            try:
                ilp_val = float(json.loads(ilp_json.read_text())["ilp_ar"])
            except Exception:
                pass
        if ilp_val is None:
            ilp_val = _load_ilp(g)
        ilp_data = {g: ilp_val}
        out_dir  = exp_root / g.upper() / "results" / "figures"
        print(f"  [{g.upper()}] {len(dirs)} seeds → {out_dir}")
        plot_experiment_summary(group_agg, ilp_data, out_dir)


_GROUP_MODULES = [
    "problem2_ilp", "problem2_ilp.objects",
    "problem3_ppo", "problem3_ppo.env",
    "problem4_ppo_mask", "problem4_ppo_mask.env",
    "problem5_ppo_lagrangian", "problem5_ppo_lagrangian.env",
    "problem6_ppo_opt", "problem6_ppo_opt.env",
    "problem_dqn", "problem_dqn.env",
    "problem_ddqn", "problem_ddqn.env", "problem_ddqn.run_all",
]


def _switch_group(group: str):
    """Update sys.path and evict cached group modules so imports resolve fresh."""
    new_pkg = str(GROUP_META[group]["pkg_dir"])
    # remove all other group pkg dirs from sys.path
    other_pkgs = {str(GROUP_META[g]["pkg_dir"]) for g in GROUP_META if g != group}
    sys.path[:] = [p for p in sys.path if p not in other_pkgs]
    if new_pkg not in sys.path:
        sys.path.insert(0, new_pkg)
    # evict stale cached modules so Python re-imports from the new path
    for mod in list(sys.modules.keys()):
        if any(mod == m or mod.startswith(m + ".") for m in _GROUP_MODULES):
            del sys.modules[mod]


def run_one_seed(seed: int, scenarios: list, group: str,
                 target_steps: int, outdir_root: Path,
                 run_id: str = ""):
    # ── 0. set up sys.path for this group ────────────────────────────────────
    _switch_group(group)

    (ECU, SVC, P3Env, P4Env, LagrangeEnv,
     P6Env, DQNEnv, DDQNEnv, DDQN) = _import_group_modules()

    M = len(scenarios[0][1])
    target_episodes = target_steps // M

    (EpisodeTrackingCallback, LagrangianUpdateCallback,
     evaluate_model, aggregate_eval,
     plot_training_curves, plot_test_results) = _import_experiment_modules()

    device = _resolve_device()
    outdir = outdir_root / str(seed)
    train_fig_dir  = outdir / "train_curve"  / "figures"
    train_data_dir = outdir / "train_curve"  / "data"
    test_data_dir  = outdir / "test_results" / "data"
    test_fig_dir   = outdir / "test_results" / "figures"
    logs_dir       = outdir / "logs"
    for d in [train_fig_dir, train_data_dir, test_data_dir, test_fig_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── tee stdout+stderr into logs/run.log ──────────────────────────────────
    _log_fh      = open(logs_dir / "run.log", "w", buffering=1)
    _prev_stdout = sys.stdout
    _prev_stderr = sys.stderr
    sys.stdout   = _Tee(_prev_stdout, _log_fh)
    sys.stderr   = _Tee(_prev_stderr, _log_fh)

    try:
        print(f"\n{'='*64}")
        print(f"  Experiment: group={group}  seed={seed}  device={device}")
        print(f"  M={M}  target_steps={target_steps:,}  target_episodes={target_episodes:,}  |  outdir: {outdir}")
        print(f"{'='*64}\n")

        # ── 1. shared train/test split ────────────────────────────────────────
        train_scenarios, test_scenarios = _make_split(scenarios, seed)
        print(f"  Split: {len(train_scenarios)} train / {len(test_scenarios)} test\n")

        # ── 2. train all models ───────────────────────────────────────────────
        training_data: dict = {}
        models: dict = {}

        def _record(name, model, cb):
            models[name] = model
            n = len(cb.episode_rewards)
            training_data[name] = {
                "episode_nums":    list(range(1, n + 1)),
                "episode_rewards": cb.episode_rewards,
            }

        model, cb = train_p3(ECU, SVC, P3Env, EpisodeTrackingCallback,
                             train_scenarios, seed, device, target_episodes)
        _record("P3_PPO", model, cb)

        model, cb = train_p4(ECU, SVC, P4Env, EpisodeTrackingCallback,
                             train_scenarios, seed, device, target_episodes)
        _record("P4_MaskPPO", model, cb)

        model, cb = train_p5(ECU, SVC, LagrangeEnv,
                             EpisodeTrackingCallback, LagrangianUpdateCallback,
                             train_scenarios, seed, device, target_episodes)
        _record("P5_LagPPO", model, cb)

        model, cb = train_p6(ECU, SVC, P6Env, EpisodeTrackingCallback,
                             train_scenarios, seed, device, target_episodes)
        _record("P6_RepairPPO", model, cb)

        model, cb = train_dqn(ECU, SVC, DQNEnv, EpisodeTrackingCallback,
                              train_scenarios, seed, device, target_episodes)
        _record("DQN", model, cb)

        model, cb = train_ddqn(ECU, SVC, DDQNEnv, DDQN, EpisodeTrackingCallback,
                               train_scenarios, seed, device, target_episodes)
        _record("DDQN", model, cb)

        # ── 3. training curve ─────────────────────────────────────────────────
        print("\n  Plotting training curves …")
        plot_training_curves(training_data, train_fig_dir, seed=seed)
        _save_training_csv(training_data, train_data_dir)

        # ── 4. evaluate all models ────────────────────────────────────────────
        print("\n  Evaluating on test scenarios …")
        all_eval: dict = {}

        def _ppo_fn(m):
            def fn(obs): a, _ = m.predict(obs, deterministic=True); return int(a)
            return fn

        def _mask_fn(m):
            def fn(obs, mask): a, _ = m.predict(obs, action_masks=mask, deterministic=True); return int(a)
            return fn

        eval_cfg = {
            "P3_PPO":       (P3Env,       _ppo_fn,  False, {}),
            "P4_MaskPPO":   (P4Env,       _mask_fn, True,  {}),
            "P5_LagPPO":    (LagrangeEnv, _ppo_fn,  False, {"lambda_init": 0.0, "lambda_max": 5.0}),
            "P6_RepairPPO": (P6Env,       _ppo_fn,  False, {}),
            "DQN":          (DQNEnv,      _ppo_fn,  False, {}),
            "DDQN":         (DDQNEnv,     _ppo_fn,  False, {}),
        }

        for name in MODELS_ORDERED:
            env_cls, policy_builder, needs_mask, env_kw = eval_cfg[name]
            results = evaluate_model(
                policy_fn=policy_builder(models[name]),
                test_scenarios=test_scenarios,
                env_cls=env_cls,
                env_kwargs=env_kw,
                needs_mask=needs_mask,
            )
            all_eval[name] = results
            agg = aggregate_eval(results)
            print(f"    {name:<16} AR={agg['ar_mean']:.4f}  "
                  f"success={agg['success_rate']:.1%}  "
                  f"cap={agg['cap_viol_rate']:.1%}  conf={agg['conf_viol_rate']:.1%}")

        # ── 5. test result plots + CSVs ───────────────────────────────────────
        agg_all = {name: aggregate_eval(all_eval[name]) for name in MODELS_ORDERED}
        plot_test_results(agg_all, test_fig_dir, seed=seed)
        _save_test_csv(all_eval, test_data_dir)
        _save_agg_csv(agg_all, test_data_dir)

        print(f"\n  Done seed={seed}. Outputs → {outdir}\n")

    finally:
        sys.stdout = _prev_stdout
        sys.stderr = _prev_stderr
        _log_fh.close()


# ══════════════════════════════════════════════════════════════════════════════
#  entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, nargs="+", default=[1])
    parser.add_argument("--group",  type=str, default="all",
                        choices=["eq", "lt", "gt", "all"])
    parser.add_argument("--steps",    type=int, default=TARGET_STEPS)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Target episodes per model (overrides --steps; steps = episodes * M per group).")
    parser.add_argument("--name",   type=str, default="ecu_exp",
                        help="Experiment name prefix for the reports directory.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Override output root (skips reports/ layout).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Shared run tag (YYYYMMDD_HHMMSS_8hex). "
                             "Auto-generated if omitted.")
    args = parser.parse_args()

    if args.run_id:
        run_id = args.run_id
        hash8  = run_id.split("_")[-1]
    else:
        hash8  = os.urandom(4).hex()
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + hash8
    ts = "_".join(run_id.split("_")[:2]) if run_id.count("_") >= 2 else ""
    print(f"  run_id : {run_id}  (hash={hash8})")

    groups = list(GROUP_META.keys()) if args.group == "all" else [args.group]

    # Determine experiment root directory
    if args.outdir:
        exp_root = Path(args.outdir)
    else:
        exp_root = ROOT / "reports" / f"{args.name}_{run_id}"

    exp_root.mkdir(parents=True, exist_ok=True)
    print(f"  exp_root: {exp_root}")
    run_info = {
        "name":      args.name,
        "hash":      hash8,
        "timestamp": ts,
        "run_id":    run_id,
        "groups":    groups,
        "seeds":     args.seed,
    }
    info_path = exp_root / "run_info.json"
    if not info_path.exists():
        info_path.write_text(json.dumps(run_info, indent=2))

    # ── parallel group execution when --group all ─────────────────────────────
    if args.group == "all":
        import subprocess as _sp
        python = sys.executable
        procs: list[tuple[str, "_sp.Popen[bytes]"]] = []
        for g in groups:
            # compute per-group steps so every group gets the same episode count
            g_scenarios = _load_scenarios(g)
            g_M = len(g_scenarios[0][1])
            if args.episodes is not None:
                g_steps = args.episodes * g_M
            else:
                g_steps = (args.steps // g_M) * g_M
            cmd = [
                python, str(Path(__file__).resolve()),
                "--group", g,
                "--run-id", run_id,
                "--outdir", str(exp_root),
                "--steps", str(g_steps),
                "--name", args.name,
                "--seed", *[str(s) for s in args.seed],
            ]
            log_path = exp_root / g.upper() / "group.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_path, "w", buffering=1)
            p = _sp.Popen(cmd, stdout=fh, stderr=fh)
            procs.append((g, p, fh))
            print(f"  [{g.upper()}] spawned PID={p.pid}  log → {log_path}")

        for g, p, fh in procs:
            rc = p.wait()
            fh.close()
            print(f"  [{g.upper()}] finished  rc={rc}")

        _generate_summary(exp_root, groups)
        print("All done.")
        return

    # ── single-group execution (also used by parallel sub-processes) ──────────
    for group in groups:
        scenarios   = _load_scenarios(group)
        outdir_root = exp_root / group.upper() / "seeds"
        group_M     = len(scenarios[0][1])
        target_steps = args.episodes * group_M if args.episodes is not None else args.steps
        print(f"\n{'='*60}\n  Group: {group.upper()}\n{'='*60}")
        _compute_and_save_ilp(group, scenarios, exp_root)
        for seed in args.seed:
            run_one_seed(
                seed=seed,
                scenarios=scenarios,
                group=group,
                target_steps=target_steps,
                outdir_root=outdir_root,
                run_id=run_id,
            )

    # ── cross-seed summary figures (only for reports/ layout) ────────────────
    if not args.outdir:
        _generate_summary(exp_root, groups)

    print("All done.")


if __name__ == "__main__":
    main()
