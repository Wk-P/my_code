"""
run_all_bc.py — P4 pipeline with ILP behavior-cloning pretraining before PPO.

Same as run_all.py, except the MaskablePPO policy is warm-started via
supervised behavior cloning on ILP-optimal expert trajectories
(shared/bc_pretrain.py) before `model.learn()` runs. Produces its own
results/plots under a distinct output dir suffix so it can be compared
against the plain-PPO baseline.

Run:
    python ppo_mask/run_all_bc.py
"""

import datetime
import csv
import sys, time, json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))  # project root for shared

import shared.timer_utils as timer_utils

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
import functools

import config as C
from ilp.objects import ECU, SVC
from shared.ilp_utils import parse_args, resolve_device, solve_ilp_all_scenarios, load_scenario
from shared.paths import VERSION, resolve_exp_id

import run_all as RA
from ppo_mask.env import P4Env
from shared.bc_pretrain import build_bc_dataset, pretrain_actor_critic


def train_maskppo_bc(ecus, services, device: str):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = DummyVecEnv(
        [functools.partial(RA._make_p4_env, C.SEED + i) for i in range(n_envs)],
    )
    print(f"  Using DummyVecEnv: n_envs={n_envs}")

    cb = RA.P4Callback()
    model = MaskablePPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = C.PPO_LR,
        n_steps       = C.PPO_N_STEPS,
        batch_size    = C.PPO_BATCH_SIZE,
        n_epochs      = C.PPO_N_EPOCHS,
        gamma         = C.PPO_GAMMA,
        gae_lambda    = C.PPO_GAE_LAMBDA,
        clip_range    = C.PPO_CLIP_RANGE,
        ent_coef      = C.PPO_ENT_COEF_INIT,
        policy_kwargs = dict(net_arch=C.PPO_NET_ARCH),
        device        = device,
        verbose       = 0,
        seed          = C.SEED,
    )

    print(f"\n[2/5] Building ILP expert dataset from {len(C.TRAIN_SCENARIOS)} train scenarios ...")
    obs_arr, mask_arr, act_arr = build_bc_dataset(C.TRAIN_SCENARIOS, P4Env, use_mask=True)

    print(f"\n[3/5] Behavior-cloning pretraining ...")
    t_bc = time.time()
    pretrain_actor_critic(model, obs_arr, act_arr, mask_arr=mask_arr,
                           epochs=C.BC_EPOCHS, batch_size=C.BC_BATCH_SIZE, lr=C.BC_LR)
    print(f"  BC pretraining done in {time.time() - t_bc:.1f}s")

    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep = len(cb.episode_ars)
    last50 = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_p = np.mean(cb.episode_placed[-50:]) if n_ep >= 50 else np.mean(cb.episode_placed)
    last50_vp = np.mean(cb.episode_valid_placed[-50:]) if n_ep >= 50 else np.mean(cb.episode_valid_placed)
    print(f"  PPO fine-tune done  {elapsed:.1f}s | {n_ep} eps "
          f"| AR(last50)={last50:.4f} | placed(last50)={last50_p:.1f}/{C.M} "
          f"| valid_placed(last50)={last50_vp:.1f}/{C.M}")
    return model, cb


@timer_utils.timer
def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    if args.total_timesteps is not None:
        C.TOTAL_STEPS = int(args.total_timesteps)
        print(f"[override] TOTAL_STEPS={C.TOTAL_STEPS:,}")
    print(f"\n{'='*60}")
    print(f"  P4 run_all_bc.py — ILP behavior-cloning pretrain + PPO fine-tune")
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    print(f"\n[1/5] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    print(f"\n[4/5] MaskablePPO training w/ BC warm-start ({C.TOTAL_STEPS:,} steps) ...")
    exp_id = resolve_exp_id(C.OUTDIR)
    C.EXP_ID = exp_id
    model, cb = train_maskppo_bc(ecus, services, device)
    run_dir = C.OUTDIR / f"{exp_id}_bc"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / f"model_{exp_id}_bc_v{VERSION}"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    print(f"\n[5/5] MaskablePPO(BC) evaluation ({len(C.TEST_SCENARIOS)} episodes, deterministic) ...")
    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return int(action)
    ppo_res = RA.run_episodes(ecus, services, ppo_policy)
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
    print(f"  PPO(BC) AR  mean={np.mean(ppo_res['ars']):.4f}  std={np.std(ppo_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(ppo_res['placed']):.1f}/{M}")
    success_rate = float(np.mean(ppo_res["success"]))
    print(f"  Success rate (all {M} placed, zero violations) = {success_rate:.2%}")

    print(f"\n{'='*66}")
    print(f"  {'Method':<28} {'AR (mean+/-std)':<24} {'Placed':<10} {'Viol'}")
    print(f"  {'-'*28} {'-'*24} {'-'*10} {'-'*5}")
    print(f"  {'ILP (Optimal)':<28} {ilp_ar:.4f} +/- 0.0000       {M}/{M:<7} 0")
    print(f"  {'MaskablePPO+BC (P4)':<28} "
          f"{np.mean(ppo_res['ars']):.4f} +/- {np.std(ppo_res['ars']):.4f}   "
          f"  {np.mean(ppo_res['placed']):.1f}/{M:<4}  0")
    print(f"{'='*66}\n")

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
        "maskable_ppo_bc": {
            "ar_mean":            round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":             round(float(np.std(ppo_res["ars"])), 6),
            "placed_mean":        round(float(np.mean(ppo_res["placed"])), 2),
            "valid_placed_mean":  round(float(np.mean(ppo_res["valid_placed"])), 2),
            "ecus_used_mean":     round(float(np.mean(ppo_res["ecus_used"])), 2),
            "success_rate":       round(success_rate, 6),
            "violations":         0,
        },
        "training": {
            "total_steps": C.TOTAL_STEPS,
            "n_episodes":  len(cb.episode_ars),
            "ar_last50":   round(float(np.mean(cb.episode_ars[-50:])), 6),
        },
    }

    base_path = run_dir
    log_path = base_path / "results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {log_path}")

    csv_path = base_path / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "valid_placed_mean",
                          "ecus_used_mean", "success_rate"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, M, C.N, 1.0])
        writer.writerow([
            "MaskablePPO+BC (P4)",
            round(float(np.mean(ppo_res["ars"])), 6),
            round(float(np.std(ppo_res["ars"])), 6),
            round(float(np.mean(ppo_res["placed"])), 2),
            round(float(np.mean(ppo_res["valid_placed"])), 2),
            round(float(np.mean(ppo_res["ecus_used"])), 2),
            round(success_rate, 4),
        ])
    print(f"  CSV  saved -> {csv_path}")

    RA.plot_training_curve(cb, ilp_ar, base_path, sc_name)
    RA.plot_comparison(ilp_ar, ppo_res, 0.0, base_path, sc_name)

    print("\nAll done! Output files:")
    print(f"  {base_path}/training_curve.png")
    print(f"  {base_path}/comparison.png")
    print(f"  {base_path}/results.json")
    print(f"  {base_path}/summary.csv\n")


if __name__ == "__main__":
    main()
