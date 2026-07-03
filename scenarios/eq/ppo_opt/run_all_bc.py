"""
run_all_bc.py — P6 pipeline (PPO + best-fit repair) with ILP behavior-cloning
pretraining. Same as run_all.py, except the plain-PPO policy is warm-started
via supervised behavior cloning on ILP-optimal expert trajectories
(shared/bc_pretrain.py) before `model.learn()` runs. No action masking is
used (matches P6 design; invalid picks get auto-repaired).

Run:
    python ppo_opt/run_all_bc.py
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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import config as C
from ppo_opt.env import P6Env
from ilp.objects import ECU, SVC
from shared.ilp_utils import parse_args, resolve_device, solve_ilp_all_scenarios, load_scenario
from shared.paths import VERSION, resolve_exp_id
from shared.bc_pretrain import build_bc_dataset, pretrain_actor_critic

import run_all as RA


def train_ppo_bc(ecus, services, device: str):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = DummyVecEnv(
        [functools.partial(RA._make_p6_env, C.SEED + i) for i in range(n_envs)],
    )
    print(f"  Using DummyVecEnv: n_envs={n_envs}")
    cb = RA.P6Callback()
    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = C.PPO_LR,
        n_steps       = C.PPO_N_STEPS,
        batch_size    = C.PPO_BATCH_SIZE,
        n_epochs      = C.PPO_N_EPOCHS,
        gamma         = C.PPO_GAMMA,
        gae_lambda    = C.PPO_GAE_LAMBDA,
        clip_range    = C.PPO_CLIP_RANGE,
        ent_coef      = C.PPO_ENT_COEF,
        policy_kwargs = dict(net_arch=C.PPO_NET_ARCH),
        device        = device,
        verbose       = 0,
        seed          = C.SEED,
    )

    print(f"\n[2/5] Building ILP expert dataset from {len(C.TRAIN_SCENARIOS)} train scenarios ...")
    obs_arr, _mask_arr, act_arr = build_bc_dataset(C.TRAIN_SCENARIOS, P6Env, use_mask=False)

    print(f"\n[3/5] Behavior-cloning pretraining ...")
    t_bc = time.time()
    pretrain_actor_critic(model, obs_arr, act_arr, mask_arr=None,
                           epochs=C.BC_EPOCHS, batch_size=C.BC_BATCH_SIZE, lr=C.BC_LR)
    print(f"  BC pretraining done in {time.time() - t_bc:.1f}s")

    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep   = len(cb.episode_ars)
    last50 = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    print(f"  PPO fine-tune done  {elapsed:.1f}s | {n_ep} episodes | AR(last50)={last50:.4f}")
    return model, cb


@timer_utils.timer
def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    if args.total_timesteps is not None:
        C.TOTAL_STEPS = int(args.total_timesteps)
        print(f"[override] TOTAL_STEPS={C.TOTAL_STEPS:,}")
    print(f"\n{'='*60}")
    print(f"  P6 run_all_bc.py — ILP behavior-cloning pretrain + PPO+repair fine-tune")
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    print(f"\n[1/5] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    print(f"\n[4/5] PPO+repair training w/ BC warm-start ({C.TOTAL_STEPS:,} steps) ...")
    exp_id = resolve_exp_id(C.OUTDIR)
    C.EXP_ID = exp_id
    model, cb = train_ppo_bc(ecus, services, device)
    run_dir = C.OUTDIR / f"{exp_id}_bc"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"model_{exp_id}_bc_v{VERSION}"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    print(f"\n[5/5] PPO+repair(BC) evaluation ({len(C.TEST_SCENARIOS)} episodes, deterministic) ...")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ppo_res = RA.run_episodes(ecus, services, ppo_policy)
    print(f"  PPO(BC) AR  mean={np.mean(ppo_res['ars']):.4f}  std={np.std(ppo_res['ars']):.4f}")
    print(f"  Eval viol rate mean={np.mean(ppo_res['repair_rates']):.2%}")

    p_v = float(np.mean(ppo_res["repair_rates"]))
    p_v_std = float(np.std(ppo_res["repair_rates"]))
    success_rate = float(np.mean(ppo_res["success"]))
    cap_viol_rate = float(np.mean(ppo_res["cap_violations"] > 0))
    conflict_viol_rate = float(np.mean(ppo_res["conflict_violations"] > 0))

    print(f"\n{'='*62}")
    print(f"  {'Method':<24} {'AR (mean±std)':<22} {'ViolRate':<10}")
    print(f"  {'-'*24} {'-'*22} {'-'*10}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} ± 0.0000       {'0':<10}")
    print(f"  {'PPO+BC (P6, repair)':<24} "
          f"{np.mean(ppo_res['ars']):.4f} ± {np.std(ppo_res['ars']):.4f}   "
          f"  {p_v:<10.2%}")
    print(f"  Success rate (all {M} placed, zero violations) = {success_rate:.2%}")
    print(f"{'='*62}\n")

    log = {
        "created_at": datetime.datetime.now().isoformat(),
        "exp_id": exp_id,
        "scenario": sc_name,
        "prototype_scenario": prototype_name,
        "scenario_count": len(C.SCENARIOS),
        "train_count": len(C.TRAIN_SCENARIOS),
        "test_count": len(C.TEST_SCENARIOS),
        "N": N, "M": M,
        "bc": {"epochs": C.BC_EPOCHS, "batch_size": C.BC_BATCH_SIZE, "lr": C.BC_LR},
        "ilp": {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "ppo_bc": {
            "ar_mean":            round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":             round(float(np.std(ppo_res["ars"])),  6),
            "viol_rate_mean":     round(float(p_v), 6),
            "success_rate":       round(success_rate, 6),
            "cap_viol_rate":      round(cap_viol_rate, 6),
            "conflict_viol_rate": round(conflict_viol_rate, 6),
            "cap_viol_total":     int(np.sum(ppo_res["cap_violations"])),
            "conflict_viol_total": int(np.sum(ppo_res["conflict_violations"])),
        },
        "training": {
            "total_steps":  C.TOTAL_STEPS,
            "n_episodes":   len(cb.episode_ars),
            "ar_last50":    round(float(np.mean(cb.episode_ars[-50:])), 6),
            "repair_rate_last50": round(float(np.mean(cb.episode_repair_rates[-50:])), 6),
        }
    }
    log_path = run_dir / "results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {log_path}")

    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "valid_placed_mean", "ecus_used_mean", "success_rate", "cap_viol_rate", "conflict_viol_rate", "cap_viol_total", "conflict_viol_total"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, M, C.N, 1.0, 0.0, 0.0, 0, 0])
        writer.writerow([
            "PPO+BC (P6, best-fit)",
            round(float(np.mean(ppo_res["ars"])), 6),
            round(float(np.std(ppo_res["ars"])), 6),
            round(float(np.mean(ppo_res["placed"])), 2),
            round(float(np.mean(ppo_res["valid_placed"])), 2),
            round(float(np.mean(ppo_res["ecus_used"])), 2),
            round(success_rate, 4),
            round(cap_viol_rate, 4),
            round(conflict_viol_rate, 4),
            int(np.sum(ppo_res["cap_violations"])),
            int(np.sum(ppo_res["conflict_violations"])),
        ])
    print(f"  CSV  saved -> {csv_path}")
    RA.plot_training_curve(cb, ilp_ar, run_dir, sc_name)
    RA.plot_comparison(ilp_ar, ppo_res, p_v, p_v_std, run_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {run_dir}/training_curve.png")
    print(f"  {run_dir}/comparison.png")
    print(f"  {run_dir}/results.json")
    print(f"  {run_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
