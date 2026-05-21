"""
Unified experiment runner — one seed, one group (eq / lt / gt), all 6 models.

Supports all three scenario groups by dynamically resolving the correct package
directory and YAML config for each group.

Usage:
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
TARGET_EPISODES = 2_000_000
_MAX_TIMESTEPS  = 999_999_999
TORCH_THREADS   = 8


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
    from problem7_ppo_seq.env           import P7Env
    return ECU, SVC, P3Env, P4Env, LagrangeEnv, P6Env, DQNEnv, DDQNEnv, DDQN, P7Env


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
    print("  [P3 PPO] training …")
    t0 = time.time()
    ep_cb = EpisodeTrackingCallback(target_episodes)
    vec   = _make_vecenv([
        functools.partial(_make_env, P3Env, ECU, SVC, seed + i, train_scenarios)
        for i in range(40)
    ])
    model = _train_ppo(vec, ep_cb, [], seed, device, dict(
        learning_rate=3e-4, n_steps=256, batch_size=128, n_epochs=10,
        gamma=0.999, gae_lambda=0.95, clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
    ))
    print(f"  [P3 PPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def train_p4(ECU, SVC, P4Env, EpisodeTrackingCallback,
             train_scenarios, seed, device, target_episodes):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    print("  [P4 MaskPPO] training …")
    t0 = time.time()
    torch.set_num_threads(TORCH_THREADS)
    ep_cb = EpisodeTrackingCallback(target_episodes)
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
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=ep_cb)
    vec.close()
    print(f"  [P4 MaskPPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def _make_p7_env(P7Env, ECU, SVC, seed_i, train_scenarios):
    from stable_baselines3.common.monitor import Monitor
    from sb3_contrib.common.wrappers import ActionMasker
    random.seed(seed_i)
    caps, reqs, _ = train_scenarios[0]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env  = P7Env(ecus, svcs, scenarios=train_scenarios)
    env  = ActionMasker(env, lambda e: e.action_masks())
    return Monitor(env)


def train_p7(ECU, SVC, P7Env, EpisodeTrackingCallback,
             train_scenarios, seed, device, target_episodes):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    print("  [P7 SeqPPO] training …")
    t0 = time.time()
    torch.set_num_threads(TORCH_THREADS)
    ep_cb = EpisodeTrackingCallback(target_episodes)
    vec   = DummyVecEnv([
        functools.partial(_make_p7_env, P7Env, ECU, SVC, seed + i, train_scenarios)
        for i in range(40)
    ])
    # Larger network to handle bigger action space (M×N)
    model = MaskablePPO(
        policy="MlpPolicy", env=vec, learning_rate=3e-4, n_steps=256,
        batch_size=128, n_epochs=10, gamma=0.999, gae_lambda=0.95,
        clip_range=0.2, policy_kwargs=dict(net_arch=[256, 256, 128]),
        device=device, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=ep_cb)
    vec.close()
    print(f"  [P7 SeqPPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def train_p5(ECU, SVC, LagrangeEnv, EpisodeTrackingCallback, LagrangianUpdateCallback,
             train_scenarios, seed, device, target_episodes):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    print("  [P5 LagPPO] training …")
    t0 = time.time()
    torch.set_num_threads(TORCH_THREADS)
    LAMBDA_INIT, LAMBDA_MAX, LAMBDA_LR, LAMBDA_TARGET, LAMBDA_WIN = 0.1, 5.0, 0.005, 0.0, 20
    ep_cb  = EpisodeTrackingCallback(target_episodes)
    lam_cb = LagrangianUpdateCallback(LAMBDA_INIT, LAMBDA_LR, LAMBDA_TARGET, LAMBDA_MAX, LAMBDA_WIN)
    vec = _make_vecenv([
        functools.partial(_make_env, LagrangeEnv, ECU, SVC, seed + i, train_scenarios,
                          {"lambda_init": LAMBDA_INIT, "lambda_max": LAMBDA_MAX})
        for i in range(40)
    ])
    model = PPO(
        policy="MlpPolicy", env=vec, learning_rate=3e-4, n_steps=128,
        batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
        device=device, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=CallbackList([ep_cb, lam_cb]))
    vec.close()
    print(f"  [P5 LagPPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} eps")
    return model, ep_cb


def train_p6(ECU, SVC, P6Env, EpisodeTrackingCallback,
             train_scenarios, seed, device, target_episodes):
    print("  [P6 RepairPPO] training …")
    t0 = time.time()
    ep_cb = EpisodeTrackingCallback(target_episodes)
    vec   = _make_vecenv([
        functools.partial(_make_env, P6Env, ECU, SVC, seed + i, train_scenarios)
        for i in range(40)
    ])
    model = _train_ppo(vec, ep_cb, [], seed, device, dict(
        learning_rate=3e-4, n_steps=256, batch_size=128, n_epochs=10,
        gamma=0.999, gae_lambda=0.95, clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
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

MODELS_ORDERED = ["P3_PPO", "P4_MaskPPO", "P5_LagPPO", "P6_RepairPPO", "DQN", "DDQN", "P7_SeqPPO"]


_GROUP_MODULES = [
    "problem2_ilp", "problem2_ilp.objects",
    "problem3_ppo", "problem3_ppo.env",
    "problem4_ppo_mask", "problem4_ppo_mask.env",
    "problem5_ppo_lagrangian", "problem5_ppo_lagrangian.env",
    "problem6_ppo_opt", "problem6_ppo_opt.env",
    "problem_dqn", "problem_dqn.env",
    "problem_ddqn", "problem_ddqn.env", "problem_ddqn.run_all",
    "problem7_ppo_seq", "problem7_ppo_seq.env",
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
                 target_episodes: int, outdir_root: Path,
                 run_id: str = ""):
    # ── 0. set up sys.path for this group ────────────────────────────────────
    _switch_group(group)

    (ECU, SVC, P3Env, P4Env, LagrangeEnv,
     P6Env, DQNEnv, DDQNEnv, DDQN, P7Env) = _import_group_modules()

    (EpisodeTrackingCallback, LagrangianUpdateCallback,
     evaluate_model, aggregate_eval,
     plot_training_curves, plot_test_results) = _import_experiment_modules()

    device = _resolve_device()
    outdir = outdir_root / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Experiment: group={group}  seed={seed}  device={device}")
    print(f"  Target episodes: {target_episodes:,}  |  outdir: {outdir}")
    print(f"{'='*64}\n")

    # ── 1. shared train/test split ────────────────────────────────────────────
    train_scenarios, test_scenarios = _make_split(scenarios, seed)
    print(f"  Split: {len(train_scenarios)} train / {len(test_scenarios)} test\n")

    # ── 2. train all models ───────────────────────────────────────────────────
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

    model, cb = train_p7(ECU, SVC, P7Env, EpisodeTrackingCallback,
                         train_scenarios, seed, device, target_episodes)
    _record("P7_SeqPPO", model, cb)

    # ── 3. training curve ─────────────────────────────────────────────────────
    print("\n  Plotting training curves …")
    plot_training_curves(training_data, outdir, seed=seed)
    _save_training_csv(training_data, outdir)

    # ── 4. evaluate all models ────────────────────────────────────────────────
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
        "P7_SeqPPO":    (P7Env,       _mask_fn, True,  {}),
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

    # ── 5. test result plots + CSVs ───────────────────────────────────────────
    agg_all = {name: aggregate_eval(all_eval[name]) for name in MODELS_ORDERED}
    plot_test_results(agg_all, outdir, seed=seed)
    _save_test_csv(all_eval, outdir)
    _save_agg_csv(agg_all, outdir)

    # ── 6. stamp run_id so plot_metrics.py can group this batch ──────────────
    if run_id:
        (outdir / ".run_id").write_text(run_id + "\n")

    print(f"\n  Done seed={seed}. Outputs → {outdir}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, nargs="+", default=[1])
    parser.add_argument("--group",    type=str, default="all",
                        choices=["eq", "lt", "gt", "all"])
    parser.add_argument("--episodes", type=int, default=TARGET_EPISODES)
    parser.add_argument("--outdir",   type=str, default=None)
    parser.add_argument("--run-id",   type=str, default=None,
                        help="Shared experiment tag (timestamp_8hex). "
                             "Auto-generated if omitted. Pass the same value "
                             "to all parallel group launches via run_all_parallel.py.")
    args = parser.parse_args()

    run_id = args.run_id or (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(4).hex()
    )
    print(f"  run_id: {run_id}")

    groups = list(GROUP_META.keys()) if args.group == "all" else [args.group]

    for group in groups:
        scenarios   = _load_scenarios(group)
        outdir_root = Path(args.outdir) / group if args.outdir else ROOT / "results" / group
        print(f"\n{'='*60}\n  Group: {group.upper()}\n{'='*60}")
        for seed in args.seed:
            run_one_seed(
                seed=seed,
                scenarios=scenarios,
                group=group,
                target_episodes=args.episodes,
                outdir_root=outdir_root,
                run_id=run_id,
            )

    print("All done.")


if __name__ == "__main__":
    main()
