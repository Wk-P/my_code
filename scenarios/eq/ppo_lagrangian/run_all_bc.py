"""
run_all_bc.py — P5 pipeline (Lagrangian PPO) with ILP behavior-cloning pretraining.

Same as run_all.py, except the plain-PPO policy is warm-started via supervised
behavior cloning on ILP-optimal expert trajectories (shared/bc_pretrain.py)
before `model.learn()` runs. No action masking is used (matches P5 design);
BC transitions are filtered to zero-violation expert replays only.

Run:
    python ppo_lagrangian/run_all_bc.py
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
from ppo_lagrangian.env import LagrangeEnv
from ilp.objects import ECU, SVC
from shared.ilp_utils import parse_args, resolve_device, solve_ilp_all_scenarios, load_scenario
from shared.paths import VERSION, resolve_exp_id
from shared.bc_pretrain import build_bc_dataset, pretrain_actor_critic

import run_all as RA


def train_lagrange_bc(device: str, n_envs: int = 1):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    env_fns = [functools.partial(RA._make_lagrange_env, C.SEED + i) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    print(f"  Using DummyVecEnv: n_envs={n_envs}")
    cb = RA.LagrangeCallback()

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
        policy_kwargs = dict(net_arch=C.PPO_NET_ARCH),
        device        = device,
        verbose       = 0,
        seed          = C.SEED,
    )

    print(f"\n[2/5] Building ILP expert dataset from {len(C.TRAIN_SCENARIOS)} train scenarios ...")
    obs_arr, _mask_arr, act_arr = build_bc_dataset(C.TRAIN_SCENARIOS, LagrangeEnv, use_mask=False)

    print(f"\n[3/5] Behavior-cloning pretraining ...")
    t_bc = time.time()
    pretrain_actor_critic(model, obs_arr, act_arr, mask_arr=None,
                           epochs=C.BC_EPOCHS, batch_size=C.BC_BATCH_SIZE, lr=C.BC_LR)
    print(f"  BC pretraining done in {time.time() - t_bc:.1f}s")

    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb, progress_bar=False)
    elapsed = time.time() - t0
    env.close()

    n_ep = len(cb.episode_ars)
    last50_ar   = np.mean(cb.episode_ars[-50:])        if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_viol = np.mean(cb.episode_viol_rates[-50:]) if n_ep >= 50 else np.mean(cb.episode_viol_rates)
    print(f"  PPO fine-tune done  {elapsed:.1f}s | {n_ep} episodes"
          f" | AR(last50)={last50_ar:.4f}"
          f" | viol(last50)={last50_viol:.4f}"
          f" | final λ={cb.lambda_val:.4f}")
    return model, cb


@timer_utils.timer
def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    if args.total_timesteps is not None:
        C.TOTAL_STEPS = int(args.total_timesteps)
        print(f"[override] TOTAL_STEPS={C.TOTAL_STEPS:,}")
    device = resolve_device(C.DEVICE)

    print(f"\n{'='*60}")
    print(f"  P5 run_all_bc.py — ILP behavior-cloning pretrain + Lagrangian PPO fine-tune")
    n_envs = max(1, int(C.N_ENVS))
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}")
    print(f"{'='*60}\n")

    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    print(f"\n[1/5] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    print(f"\n[4/5] Lagrangian PPO training w/ BC warm-start ({C.TOTAL_STEPS:,} steps, {n_envs} envs) ...")
    exp_id = resolve_exp_id(C.OUTDIR)
    C.EXP_ID = exp_id
    model, cb = train_lagrange_bc(device, n_envs)
    run_dir = C.OUTDIR / f"{exp_id}_bc"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"model_{exp_id}_bc_v{VERSION}"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    print(f"\n[5/5] Lagrangian PPO(BC) evaluation ({len(C.TEST_SCENARIOS)} episodes, deterministic) ...")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ppo_res = RA.run_episodes(ecus, services, ppo_policy, lambda_eval=cb.lambda_val)
    # AR is only meaningful as "solution quality" for episodes that actually
    # placed everything legally — a partial/broken episode's AR isn't a
    # comparable data point against ILP's (always-successful) AR, so it's
    # excluded from ar_mean/ar_std/box-plot rather than averaged in.
    success_mask = ppo_res["success"]
    if success_mask.any():
        ppo_res["ars"] = ppo_res["ars"][success_mask]
    else:
        print("  [warn] no successful episodes -- ar_mean/ar_std computed over 0 samples (NaN)")
        ppo_res["ars"] = ppo_res["ars"][:0]
    print(f"  PPO(BC) AR mean={np.mean(ppo_res['ars']):.4f}  "
          f"Eval viol={np.mean(ppo_res['viol_rates']):.2%}")

    ppo_train_viol = float(np.mean(ppo_res["viol_rates"]))
    ppo_train_viol_std = float(np.std(ppo_res["viol_rates"]))
    success_rate = float(np.mean(ppo_res["success"]))
    cap_viol_rate = float(np.mean(ppo_res["cap_viols"] > 0))
    conflict_viol_rate = float(np.mean(ppo_res["conflict_viols"] > 0))

    print(f"\n{'='*72}")
    print(f"  {'Method':<28} {'AR mean±std':<24} {'Viol%':<14} {'Placed'}")
    print(f"  {'-'*28} {'-'*24} {'-'*14} {'-'*8}")
    print(f"  {'ILP (Optimal)':<28} {ilp_ar:.4f} ± 0.0000        {'0.00%':<14} {M}/{M}")
    print(f"  {'Lagrange PPO+BC':<28} "
          f"{np.mean(ppo_res['ars']):.4f} ± {np.std(ppo_res['ars']):.4f}   "
          f"  {ppo_train_viol:<12.2%} {M}/{M}")
    print(f"  Final λ = {cb.lambda_val:.4f}")
    print(f"{'='*72}\n")

    n_ep = len(cb.episode_ars)
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
        "lagrange_ppo_bc": {
            "ar_mean":            round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":             round(float(np.std(ppo_res["ars"])), 6),
            "viol_rate_mean":     round(float(ppo_train_viol), 6),
            "success_rate":       round(success_rate, 6),
            "cap_viol_rate":      round(cap_viol_rate, 6),
            "conflict_viol_rate": round(conflict_viol_rate, 6),
            "cap_viol_total":     int(np.sum(ppo_res["cap_viols"])),
            "conflict_viol_total": int(np.sum(ppo_res["conflict_viols"])),
        },
        "training": {
            "total_steps":    C.TOTAL_STEPS,
            "n_episodes":     n_ep,
            "ar_last50":      round(float(np.mean(cb.episode_ars[-50:])),        6) if n_ep >= 50 else None,
            "viol_last50":    round(float(np.mean(cb.episode_viol_rates[-50:])), 6) if n_ep >= 50 else None,
            "final_lambda":   round(float(cb.lambda_val), 6),
        },
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {run_dir / 'results.json'}")

    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "valid_placed_mean", "ecus_used_mean", "success_rate", "cap_viol_rate", "conflict_viol_rate", "cap_viol_total", "conflict_viol_total"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, M, C.N, 1.0, 0.0, 0.0, 0, 0])
        writer.writerow([
            "Lagrange PPO+BC",
            round(float(np.mean(ppo_res["ars"])), 6),
            round(float(np.std(ppo_res["ars"])), 6),
            round(float(np.mean(ppo_res["placed"])), 2),
            round(float(np.mean(ppo_res["valid_placed"])), 2),
            round(float(np.mean(ppo_res["ecus_used"])), 2),
            round(success_rate, 4),
            round(cap_viol_rate, 4),
            round(conflict_viol_rate, 4),
            int(np.sum(ppo_res["cap_viols"])),
            int(np.sum(ppo_res["conflict_viols"])),
        ])
    print(f"  CSV  saved -> {csv_path}")

    RA.plot_training_curve(cb, ilp_ar, run_dir, sc_name)
    RA.plot_comparison(ilp_ar, ppo_res, ppo_train_viol, ppo_train_viol_std, run_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {run_dir}/training_curve.png")
    print(f"  {run_dir}/comparison.png")
    print(f"  {run_dir}/results.json")
    print(f"  {run_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
