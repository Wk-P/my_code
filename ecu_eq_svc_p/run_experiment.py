"""
Unified experiment runner — one seed, one group (eq / lt / gt), all 6 models.

Usage:
    python run_experiment.py --seed 1 --group eq
    python run_experiment.py --seed 1 2 --group eq          # multiple seeds sequentially

Outputs (per seed):
    results/<group>/seed_<N>/
        training_curve.png     — all models' reward vs episode on one figure
        test_results.png       — AR / success-failure / violation subplots
        training_rewards.csv   — per-episode rewards for each model
        test_results.csv       — per-scenario test metrics for each model
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ── path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from problem2_ilp.objects import ECU, SVC
from problem3_ppo.env       import P3Env
from problem4_ppo_mask.env  import P4Env
from problem5_ppo_lagrangian.env import LagrangeEnv
from problem6_ppo_opt.env   import P6Env
from problem_dqn.env        import DQNEnv
from problem_ddqn.env       import DDQNEnv
from problem_ddqn.run_all   import DDQN  # Double-DQN class

from experiment.callbacks  import EpisodeTrackingCallback, LagrangianUpdateCallback
from experiment.eval_utils import evaluate_model, aggregate_eval
from experiment.plots      import plot_training_curves, plot_test_results

# ── global training target ────────────────────────────────────────────────────
TARGET_EPISODES = 2_000_000
# upper-bound timestep budget per model (callback stops training well before this)
_MAX_TIMESTEPS  = 999_999_999

TORCH_THREADS = 8


# ══════════════════════════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_split(scenarios: list, seed: int, train_ratio: float = 0.8):
    rng  = random.Random(seed)
    idxs = list(range(len(scenarios)))
    rng.shuffle(idxs)
    n_train = int(train_ratio * len(idxs))
    train   = [scenarios[i] for i in idxs[:n_train]]
    test    = [scenarios[i] for i in idxs[n_train:]]
    return train, test


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_ecus_svcs(scenario):
    caps, reqs, _ = scenario
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return ecus, svcs


# ══════════════════════════════════════════════════════════════════════════════
#  per-model environment factories
# ══════════════════════════════════════════════════════════════════════════════

def _p3_env_fn(seed_i: int, train_scenarios: list) -> Monitor:
    random.seed(seed_i)
    caps, reqs, _ = train_scenarios[0]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(P3Env(ecus, svcs, scenarios=train_scenarios))


def _p4_mask_fn(env) -> np.ndarray:
    return env.action_masks()


def _p4_env_fn(seed_i: int, train_scenarios: list) -> Monitor:
    random.seed(seed_i)
    caps, reqs, _ = train_scenarios[0]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env  = P4Env(ecus, svcs, scenarios=train_scenarios)
    env  = ActionMasker(env, _p4_mask_fn)
    return Monitor(env)


def _p5_env_fn(seed_i: int, train_scenarios: list,
               lambda_init: float, lambda_max: float) -> Monitor:
    random.seed(seed_i)
    caps, reqs, _ = train_scenarios[0]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(LagrangeEnv(ecus, svcs, scenarios=train_scenarios,
                               lambda_init=lambda_init, lambda_max=lambda_max))


def _p6_env_fn(seed_i: int, train_scenarios: list) -> Monitor:
    random.seed(seed_i)
    caps, reqs, _ = train_scenarios[0]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(P6Env(ecus, svcs, scenarios=train_scenarios))


def _dqn_env_fn(seed_i: int, train_scenarios: list, env_cls) -> Monitor:
    random.seed(seed_i)
    caps, reqs, _ = train_scenarios[0]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(env_cls(ecus, svcs, scenarios=train_scenarios))


# ══════════════════════════════════════════════════════════════════════════════
#  per-model train functions  → (model, EpisodeTrackingCallback)
# ══════════════════════════════════════════════════════════════════════════════

def _train_ppo(
    env_fn,
    n_envs: int,
    base_seed: int,
    device: str,
    target_episodes: int,
    ppo_kwargs: dict,
    extra_callbacks=None,
) -> tuple:
    torch.set_num_threads(TORCH_THREADS)
    vec_env = DummyVecEnv(
        [functools.partial(env_fn, base_seed + i) for i in range(n_envs)]
    )
    ep_cb = EpisodeTrackingCallback(target_episodes)
    cbs   = [ep_cb] + (extra_callbacks or [])
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        verbose=0,
        seed=base_seed,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=CallbackList(cbs))
    vec_env.close()
    return model, ep_cb


def train_p3(train_scenarios, seed, device, target_episodes):
    print(f"  [P3 PPO] training …")
    t0 = time.time()
    model, cb = _train_ppo(
        env_fn=functools.partial(_p3_env_fn, train_scenarios=train_scenarios),
        n_envs=40,
        base_seed=seed,
        device=device,
        target_episodes=target_episodes,
        ppo_kwargs=dict(
            learning_rate=3e-4,
            n_steps=256,
            batch_size=128,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[256, 256]),
        ),
    )
    print(f"  [P3 PPO] done  {time.time()-t0:.1f}s | {cb.episode_count:,} episodes")
    return model, cb


def train_p4(train_scenarios, seed, device, target_episodes):
    print(f"  [P4 MaskPPO] training …")
    t0 = time.time()
    torch.set_num_threads(TORCH_THREADS)
    vec_env = DummyVecEnv(
        [functools.partial(_p4_env_fn, seed_i=seed + i, train_scenarios=train_scenarios)
         for i in range(40)]
    )
    ep_cb = EpisodeTrackingCallback(target_episodes)
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=ep_cb)
    vec_env.close()
    print(f"  [P4 MaskPPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} episodes")
    return model, ep_cb


def train_p5(train_scenarios, seed, device, target_episodes):
    print(f"  [P5 LagPPO] training …")
    t0 = time.time()
    torch.set_num_threads(TORCH_THREADS)

    LAMBDA_INIT   = 0.1
    LAMBDA_MAX    = 5.0
    LAMBDA_LR     = 0.005
    LAMBDA_TARGET = 0.0
    LAMBDA_WINDOW = 20

    vec_env = DummyVecEnv([
        functools.partial(_p5_env_fn, seed_i=seed + i, train_scenarios=train_scenarios,
                          lambda_init=LAMBDA_INIT, lambda_max=LAMBDA_MAX)
        for i in range(40)
    ])
    ep_cb  = EpisodeTrackingCallback(target_episodes)
    lam_cb = LagrangianUpdateCallback(
        lambda_init=LAMBDA_INIT,
        lambda_lr=LAMBDA_LR,
        lambda_target=LAMBDA_TARGET,
        lambda_max=LAMBDA_MAX,
        update_window=LAMBDA_WINDOW,
    )
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
        device=device,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=CallbackList([ep_cb, lam_cb]))
    vec_env.close()
    print(f"  [P5 LagPPO] done  {time.time()-t0:.1f}s | {ep_cb.episode_count:,} episodes")
    return model, ep_cb


def train_p6(train_scenarios, seed, device, target_episodes):
    print(f"  [P6 RepairPPO] training …")
    t0 = time.time()
    model, cb = _train_ppo(
        env_fn=functools.partial(_p6_env_fn, train_scenarios=train_scenarios),
        n_envs=40,
        base_seed=seed,
        device=device,
        target_episodes=target_episodes,
        ppo_kwargs=dict(
            learning_rate=3e-4,
            n_steps=256,
            batch_size=128,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[256, 256]),
        ),
    )
    print(f"  [P6 RepairPPO] done  {time.time()-t0:.1f}s | {cb.episode_count:,} episodes")
    return model, cb


def _train_dqn_variant(
    model_cls,
    env_cls,
    model_name: str,
    train_scenarios,
    seed,
    device,
    target_episodes,
    dqn_kwargs: dict,
    n_envs: int = 12,
):
    torch.set_num_threads(TORCH_THREADS)
    vec_env = DummyVecEnv(
        [functools.partial(_dqn_env_fn, seed_i=seed + i,
                           train_scenarios=train_scenarios, env_cls=env_cls)
         for i in range(n_envs)]
    )
    ep_cb = EpisodeTrackingCallback(target_episodes)
    model = model_cls(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        verbose=0,
        seed=seed,
        **dqn_kwargs,
    )
    model.learn(total_timesteps=_MAX_TIMESTEPS, callback=ep_cb)
    vec_env.close()
    return model, ep_cb


_DQN_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=64,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    exploration_fraction=0.1,
    exploration_final_eps=0.0,
    policy_kwargs=dict(net_arch=[128, 128]),
)


def train_dqn(train_scenarios, seed, device, target_episodes):
    print(f"  [DQN] training …")
    t0 = time.time()
    model, cb = _train_dqn_variant(
        model_cls=DQN,
        env_cls=DQNEnv,
        model_name="DQN",
        train_scenarios=train_scenarios,
        seed=seed,
        device=device,
        target_episodes=target_episodes,
        dqn_kwargs=_DQN_KWARGS,
    )
    print(f"  [DQN] done  {time.time()-t0:.1f}s | {cb.episode_count:,} episodes")
    return model, cb


def train_ddqn(train_scenarios, seed, device, target_episodes):
    print(f"  [DDQN] training …")
    t0 = time.time()
    model, cb = _train_dqn_variant(
        model_cls=DDQN,
        env_cls=DDQNEnv,
        model_name="DDQN",
        train_scenarios=train_scenarios,
        seed=seed,
        device=device,
        target_episodes=target_episodes,
        dqn_kwargs=_DQN_KWARGS,
    )
    print(f"  [DDQN] done  {time.time()-t0:.1f}s | {cb.episode_count:,} episodes")
    return model, cb


# ══════════════════════════════════════════════════════════════════════════════
#  per-model evaluation policy wrappers
# ══════════════════════════════════════════════════════════════════════════════

def _ppo_policy(model):
    def fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return fn


def _mask_ppo_policy(model):
    def fn(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return fn


def _dqn_policy(model):
    def fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return fn


# ══════════════════════════════════════════════════════════════════════════════
#  CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_training_csv(training_data: dict[str, dict], outdir: Path) -> None:
    path = outdir / "training_rewards.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "episode", "reward"])
        for model_name, data in training_data.items():
            for ep, rew in zip(data["episode_nums"], data["episode_rewards"]):
                writer.writerow([model_name, ep, round(rew, 6)])
    print(f"  Saved → {path}")


def _save_test_csv(all_eval: dict[str, list[dict]], outdir: Path) -> None:
    path = outdir / "test_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "scenario_idx", "ar",
            "success", "services_placed", "M",
            "episode_has_cap_violation", "episode_has_conflict_violation",
        ])
        for model_name, results in all_eval.items():
            for idx, r in enumerate(results):
                writer.writerow([
                    model_name, idx,
                    round(r["ar"], 6),
                    int(r["success"]),
                    r["services_placed"],
                    r["M"],
                    int(r["episode_has_cap_violation"]),
                    int(r["episode_has_conflict_violation"]),
                ])
    print(f"  Saved → {path}")


def _save_agg_csv(agg: dict[str, dict], outdir: Path) -> None:
    path = outdir / "test_results_agg.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "ar_mean", "ar_std",
            "success_rate", "failure_rate",
            "cap_viol_rate", "conf_viol_rate",
            "placed_mean", "M", "n_scenarios",
        ])
        for model_name, d in agg.items():
            writer.writerow([
                model_name,
                round(d["ar_mean"], 6), round(d["ar_std"], 6),
                round(d["success_rate"], 6), round(d["failure_rate"], 6),
                round(d["cap_viol_rate"], 6), round(d["conf_viol_rate"], 6),
                round(d["placed_mean"], 4), d["M"], d["n_scenarios"],
            ])
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  main experiment runner (one seed)
# ══════════════════════════════════════════════════════════════════════════════

MODELS_ORDERED = ["P3_PPO", "P4_MaskPPO", "P5_LagPPO", "P6_RepairPPO", "DQN", "DDQN"]


def run_one_seed(
    seed: int,
    scenarios: list,
    group: str,
    target_episodes: int = TARGET_EPISODES,
    outdir_root: Path | None = None,
) -> None:
    device   = _resolve_device()
    outdir   = (outdir_root or HERE / "results" / group) / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Experiment: group={group}  seed={seed}  device={device}")
    print(f"  Target episodes: {target_episodes:,}  |  outdir: {outdir}")
    print(f"{'='*64}\n")

    # ── 1. shared train/test split ────────────────────────────────────────────
    train_scenarios, test_scenarios = _make_split(scenarios, seed)
    print(f"  Split: {len(train_scenarios)} train / {len(test_scenarios)} test scenarios\n")

    # ── 2. train all models ───────────────────────────────────────────────────
    training_data: dict[str, dict] = {}
    models: dict = {}

    train_fns = {
        "P3_PPO":       train_p3,
        "P4_MaskPPO":   train_p4,
        "P5_LagPPO":    train_p5,
        "P6_RepairPPO": train_p6,
        "DQN":          train_dqn,
        "DDQN":         train_ddqn,
    }

    for name in MODELS_ORDERED:
        fn = train_fns[name]
        model, cb = fn(train_scenarios, seed, device, target_episodes)
        models[name] = model
        n = len(cb.episode_rewards)
        training_data[name] = {
            "episode_nums":    list(range(1, n + 1)),
            "episode_rewards": cb.episode_rewards,
        }

    # ── 3. plot training curves ───────────────────────────────────────────────
    print("\n  Plotting training curves …")
    plot_training_curves(training_data, outdir, seed=seed)

    # ── 4. save training CSV ──────────────────────────────────────────────────
    _save_training_csv(training_data, outdir)

    # ── 5. evaluate all models ────────────────────────────────────────────────
    print("\n  Evaluating models on test scenarios …")
    all_eval: dict[str, list[dict]] = {}

    eval_cfg = {
        "P3_PPO":       (P3Env,      _ppo_policy,      False, {}),
        "P4_MaskPPO":   (P4Env,      _mask_ppo_policy, True,  {}),
        "P5_LagPPO":    (LagrangeEnv,_ppo_policy,      False,
                         {"lambda_init": 0.0, "lambda_max": 5.0}),
        "P6_RepairPPO": (P6Env,      _ppo_policy,      False, {}),
        "DQN":          (DQNEnv,     _dqn_policy,      False, {}),
        "DDQN":         (DDQNEnv,    _dqn_policy,      False, {}),
    }

    for name in MODELS_ORDERED:
        env_cls, policy_builder, needs_mask, env_kw = eval_cfg[name]
        policy_fn = policy_builder(models[name])
        results   = evaluate_model(
            policy_fn=policy_fn,
            test_scenarios=test_scenarios,
            env_cls=env_cls,
            env_kwargs=env_kw,
            needs_mask=needs_mask,
        )
        all_eval[name] = results
        agg = aggregate_eval(results)
        print(f"    {name:<16} AR={agg['ar_mean']:.4f}  "
              f"success={agg['success_rate']:.1%}  "
              f"cap_viol={agg['cap_viol_rate']:.1%}  "
              f"conf_viol={agg['conf_viol_rate']:.1%}")

    # ── 6. aggregate + plot test results ─────────────────────────────────────
    agg_results = {name: aggregate_eval(all_eval[name]) for name in MODELS_ORDERED}
    plot_test_results(agg_results, outdir, seed=seed)

    # ── 7. save test CSVs ─────────────────────────────────────────────────────
    _save_test_csv(all_eval, outdir)
    _save_agg_csv(agg_results, outdir)

    print(f"\n  Done seed={seed}. All outputs in {outdir}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  entry point
# ══════════════════════════════════════════════════════════════════════════════

def _load_scenarios(group: str) -> list:
    yaml_paths = {
        "eq": HERE / "problem2_ilp" / "config" / "config_ecu_eq_svc.yaml",
        "lt": HERE / "problem2_ilp" / "config" / "config_ecu_lt_svc.yaml",
        "gt": HERE / "problem2_ilp" / "config" / "config_ecu_gt_svc.yaml",
    }
    if group not in yaml_paths:
        raise ValueError(f"Unknown group '{group}'. Choose from: {list(yaml_paths)}")
    path = yaml_paths[group]
    if not path.exists():
        # try sibling group directories
        alt = HERE.parent / f"ecu_{group}_svc_p" / "problem2_ilp" / "config" / path.name
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Config YAML not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    scenarios = [
        (
            [ecu["capacity"]    for ecu in sc["ECUs"]],
            [svc["requirement"] for svc in sc["SVCs"]],
            sc.get("conflict_sets", []),
        )
        for sc in cfg["scenarios"]
    ]
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Run RL experiments for one group + seed(s).")
    parser.add_argument("--seed",     type=int, nargs="+", default=[1],
                        help="One or more random seeds (e.g. --seed 1 2)")
    parser.add_argument("--group",    type=str, default="eq",
                        choices=["eq", "lt", "gt"],
                        help="Scenario group: eq | lt | gt")
    parser.add_argument("--episodes", type=int, default=TARGET_EPISODES,
                        help=f"Target training episodes (default {TARGET_EPISODES:,})")
    parser.add_argument("--outdir",   type=str, default=None,
                        help="Root output directory (default: results/<group>)")
    args = parser.parse_args()

    scenarios  = _load_scenarios(args.group)
    outdir_root = Path(args.outdir) if args.outdir else HERE / "results" / args.group

    for seed in args.seed:
        run_one_seed(
            seed=seed,
            scenarios=scenarios,
            group=args.group,
            target_episodes=args.episodes,
            outdir_root=outdir_root,
        )

    print("All seeds complete.")


if __name__ == "__main__":
    main()
