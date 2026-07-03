"""
run_all_bc.py — DQN pipeline with ILP behavior-cloning pretraining.

Same as run_all.py, except the Q-network is warm-started via a DQfD-style
large-margin classification loss on ILP-optimal expert transitions
(shared/bc_pretrain.pretrain_dqn) before `model.learn()` runs. DQN has no
policy distribution to imitate directly (unlike PPO), hence the margin loss
instead of log-prob behavior cloning.

Run:
    python dqn/run_all_bc.py
"""

import datetime
import csv
import functools
import sys, time, json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))  # project root for shared

import shared.timer_utils as timer_utils

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import config as C
from dqn.env import DQNEnv
from ilp.objects import ECU, SVC
from shared.ilp_utils import parse_args, resolve_device, solve_ilp_all_scenarios, load_scenario
from shared.paths import VERSION, resolve_exp_id
from shared.bc_pretrain import build_bc_dataset, pretrain_dqn

import run_all as RA


def train_dqn_bc(ecus, services, device: str):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = DummyVecEnv(
        [functools.partial(RA._make_dqn_env, C.SEED + i) for i in range(n_envs)],
    )
    print(f"  Using DummyVecEnv: n_envs={n_envs}")
    cb = RA.DQNCallback()
    model = DQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = C.DQN_LR,
        buffer_size            = C.DQN_BUFFER_SIZE,
        learning_starts        = C.DQN_LEARNING_STARTS,
        batch_size              = C.DQN_BATCH_SIZE,
        tau                    = C.DQN_TAU,
        gamma                  = C.DQN_GAMMA,
        train_freq             = C.DQN_TRAIN_FREQ,
        gradient_steps         = C.DQN_GRADIENT_STEPS,
        target_update_interval = C.DQN_TARGET_UPDATE,
        exploration_fraction   = C.DQN_EXPLORATION_FRACTION,
        exploration_final_eps  = C.DQN_EXPLORATION_FINAL_EPS,
        policy_kwargs          = dict(net_arch=C.DQN_NET_ARCH),
        device                 = device,
        verbose                = 0,
        seed                   = C.SEED,
    )

    print(f"\n[2/5] Building ILP expert dataset from {len(C.TRAIN_SCENARIOS)} train scenarios ...")
    # this scenario's env does not sort services descending in reset()
    # (only lt's dqn/ddqn do) — sorted_desc=False keeps expert actions
    # aligned with the env's actual (unsorted) presentation order.
    obs_arr, _mask_arr, act_arr = build_bc_dataset(C.TRAIN_SCENARIOS, DQNEnv, use_mask=False, sorted_desc=False)

    print(f"\n[3/5] Behavior-cloning pretraining (Q-network, margin loss) ...")
    t_bc = time.time()
    pretrain_dqn(model, obs_arr, act_arr,
                 epochs=C.BC_EPOCHS, batch_size=C.BC_BATCH_SIZE, lr=C.BC_LR, margin=C.BC_MARGIN)
    print(f"  BC pretraining done in {time.time() - t_bc:.1f}s")

    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep     = len(cb.episode_rewards)
    last50_r = np.mean(cb.episode_rewards[-50:]) if n_ep >= 50 else np.mean(cb.episode_rewards)
    last50_ar = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_v = np.mean(cb.episode_violated[-50:]) if n_ep >= 50 else np.mean(cb.episode_violated)
    print(f"  DQN fine-tune done  {elapsed:.1f}s | {n_ep} eps "
            f"| reward(last50)={last50_r:.4f} | AR(last50)={last50_ar:.4f}"
            f" | viol_rate(last50)={last50_v:.2%}")
    return model, cb


@timer_utils.timer
def main():
    args = parse_args()
    if args.total_timesteps is not None:
        C.TOTAL_STEPS = int(args.total_timesteps)
        print(f"[override] TOTAL_STEPS={C.TOTAL_STEPS:,}")

    exp_id = resolve_exp_id(C.OUTDIR)
    C.EXP_ID = exp_id
    base_dir = C.OUTDIR / f"{exp_id}_bc"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DQN run_all_bc.py — ILP behavior-cloning pretrain + DQN fine-tune")
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    print(f"\n[1/5] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    print(f"\n[4/5] DQN training w/ BC warm-start ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_dqn_bc(ecus, services, device)
    model_path = base_dir / f"model_{exp_id}_bc_v{VERSION}"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    print(f"\n[5/5] DQN(BC) evaluation ({len(C.TEST_SCENARIOS)} episodes, deterministic) ...")
    def dqn_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    dqn_res = RA.run_episodes(ecus, services, dqn_policy)
    print(f"  DQN(BC) AR  mean={np.mean(dqn_res['ars']):.4f}  std={np.std(dqn_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(dqn_res['placed']):.1f}/{M}")
    print(f"  Eval viol rate  {np.mean(dqn_res['viols']):.2%}")

    dqn_train_v = float(np.mean(dqn_res["viols"]))
    dqn_train_v_std = float(np.std(dqn_res["viols"]))
    success_rate = float(np.mean(dqn_res["success"]))
    cap_viol_rate = float(np.mean(dqn_res["cap_viols"] > 0))
    conflict_viol_rate = float(np.mean(dqn_res["conflict_viols"] > 0))

    print(f"\n{'='*68}")
    print(f"  {'Method':<24} {'AR (mean+/-std)':<22} {'Placed':<10} {'Viol%'}")
    print(f"  {'-'*24} {'-'*22} {'-'*10} {'-'*6}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} +/- 0.0000     {M}/{M:<6} 0%")
    print(f"  {'DQN+BC (no mask)':<24} "
          f"{np.mean(dqn_res['ars']):.4f} +/- {np.std(dqn_res['ars']):.4f}   "
            f"  {np.mean(dqn_res['placed']):.1f}/{M:<2}   {dqn_train_v:.0%}")
    print(f"{'='*68}\n")

    log = {
        "created_at": datetime.datetime.now().isoformat(),
        "exp_id": exp_id,
        "scenario": sc_name,
        "prototype_scenario": prototype_name,
        "scenario_count": len(C.SCENARIOS),
        "train_count": len(C.TRAIN_SCENARIOS),
        "test_count": len(C.TEST_SCENARIOS),
        "N": N, "M": M,
        "bc": {"epochs": C.BC_EPOCHS, "batch_size": C.BC_BATCH_SIZE, "lr": C.BC_LR, "margin": C.BC_MARGIN},
        "ilp": {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "dqn_bc": {
            "ar_mean":            round(float(np.mean(dqn_res["ars"])), 6),
            "ar_std":             round(float(np.std(dqn_res["ars"])), 6),
            "placed_mean":        round(float(np.mean(dqn_res["placed"])), 2),
            "viol_rate":          round(float(dqn_train_v), 4),
            "success_rate":       round(success_rate, 6),
            "cap_viol_rate":      round(cap_viol_rate, 6),
            "conflict_viol_rate": round(conflict_viol_rate, 6),
            "cap_viol_total":     int(np.sum(dqn_res["cap_viols"])),
            "conflict_viol_total": int(np.sum(dqn_res["conflict_viols"])),
        },
        "training": {
            "total_steps":      C.TOTAL_STEPS,
            "n_episodes":       len(cb.episode_rewards),
            "ar_last50":        round(float(np.mean(cb.episode_ars[-50:])), 6),
            "reward_last50":    round(float(np.mean(cb.episode_rewards[-50:])), 6),
            "viol_rate_last50": round(float(np.mean(cb.episode_violated[-50:])), 4),
        },
    }
    log_path = base_dir / "results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {log_path}")

    csv_path = base_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "valid_placed_mean", "ecus_used_mean", "success_rate", "cap_viol_rate", "conflict_viol_rate", "cap_viol_total", "conflict_viol_total"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, M, C.N, 1.0, 0.0, 0.0, 0, 0])
        writer.writerow([
            "DQN+BC (no mask)",
            round(float(np.mean(dqn_res["ars"])), 6),
            round(float(np.std(dqn_res["ars"])), 6),
            round(float(np.mean(dqn_res["placed"])), 2),
            round(float(np.mean(dqn_res["valid_placed"])), 2),
            round(float(np.mean(dqn_res["ecus_used"])), 2),
            round(success_rate, 4),
            round(cap_viol_rate, 4),
            round(conflict_viol_rate, 4),
            int(np.sum(dqn_res["cap_viols"])),
            int(np.sum(dqn_res["conflict_viols"])),
        ])
    print(f"  CSV  saved -> {csv_path}")

    RA.plot_training_curve(cb, ilp_ar, base_dir, sc_name)
    RA.plot_comparison(ilp_ar, dqn_res, dqn_train_v, dqn_train_v_std, base_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {base_dir}/training_curve.png")
    print(f"  {base_dir}/comparison.png")
    print(f"  {base_dir}/results.json")
    print(f"  {base_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
