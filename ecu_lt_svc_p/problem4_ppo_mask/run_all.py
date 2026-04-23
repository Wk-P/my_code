"""
run_all.py — One-shot P4 full pipeline:

  1. Load scenario from YAML (same as P2/P3)
  2. Solve with ILP (PuLP)                        -> ilp_ar (optimal upper bound)
  3. Evaluate random policy (with action masking)  -> random_ars
  4. Train MaskablePPO (with action masking)       -> training curve
  5. Evaluate trained MaskablePPO                  -> ppo_ars
  6. Produce plots:
       - comparison.png     — 3-way AR box plot + services placed
       - training_curve.png — AR during training

P4 Design: constraints enforced via action masking -> 0 violations guaranteed.

Run:
    python problem4_ppo_mask/run_all.py
"""

import datetime
import csv
import argparse
import sys, time, json, functools
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import timer_utils

import torch
import yaml
import pulp
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import config as C
from problem4_ppo_mask.env import P4Env
from problem2_ilp.objects import ECU, SVC
from sb3_contrib.common.wrappers import ActionMasker
from run_utils import parse_args, resolve_device, moving_avg, solve_ilp, solve_ilp_all_scenarios, load_scenario


def _mask_fn(env) -> np.ndarray:
    return env.action_masks()


def _make_p4_env(seed: int) -> Monitor:
    """Module-level factory (picklable for SubprocVecEnv on Windows)."""
    import random
    random.seed(seed)
    caps, reqs, _ = C.SCENARIOS[C.SCENARIO_IDX]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env = P4Env(ecus, services, scenarios=C.SCENARIOS)
    env = ActionMasker(env, _mask_fn)
    return Monitor(env)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 & 5 — Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps):
    """policy_fn(obs, mask) -> int"""
    env = P4Env(ecus, services, scenarios=C.FEASIBLE_SCENARIOS)
    ars, placed_list = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            mask = env.action_masks()
            if not np.any(mask):
                break
            obs, _, done, _, info = env.step(policy_fn(obs, mask))
        ars.append(info.get("ar", 0.0))
        placed_list.append(info.get("services_placed", 0))
    return {
        "ars":    np.array(ars),
        "placed": np.array(placed_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Training
# ══════════════════════════════════════════════════════════════════════════════

class P4Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_ars:     list[float] = []
        self.episode_placed:  list[int]   = []
        self.timesteps_at_ep: list[int]   = []
        self._next_progress_step = C.PROGRESS_LOG_EVERY_STEPS
        self._t_start = 0.0

    def _on_training_start(self) -> None:
        self._t_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_ars.append(float(info.get("ar", 0.0)))
                self.episode_placed.append(int(info.get("services_placed", 0)))
                self.timesteps_at_ep.append(self.num_timesteps)

        if self.num_timesteps >= self._next_progress_step:
            elapsed = max(time.time() - self._t_start, 1e-6)
            pct = min(100.0, self.num_timesteps * 100.0 / C.TOTAL_STEPS)
            eps = len(self.episode_ars)
            sps = self.num_timesteps / elapsed
            print(
                f"  [train] step={self.num_timesteps:,}/{C.TOTAL_STEPS:,} "
                f"({pct:5.1f}%) | eps={eps} | steps/s={sps:,.0f}"
            )
            self._next_progress_step += C.PROGRESS_LOG_EVERY_STEPS
        return True


def train_maskppo(ecus, services, device: str):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = SubprocVecEnv(
        [functools.partial(_make_p4_env, C.SEED + i) for i in range(n_envs)],
        start_method=C.SUBPROC_START_METHOD,
    )
    print(f"  Using SubprocVecEnv: n_envs={n_envs}, start_method={C.SUBPROC_START_METHOD}")

    cb  = P4Callback()
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
        policy_kwargs = dict(net_arch=C.PPO_NET_ARCH),
        device        = device,
        verbose       = 0,
        seed          = C.SEED,
    )
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep   = len(cb.episode_ars)
    last50 = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_p = np.mean(cb.episode_placed[-50:]) if n_ep >= 50 else np.mean(cb.episode_placed)
    print(f"  Training done  {elapsed:.1f}s | {n_ep} eps "
          f"| AR(last50)={last50:.4f} | placed(last50)={last50_p:.1f}/{C.M}")
    return model, cb


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curve(cb, ilp_ar, outdir, scenario_name):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ts = np.array(cb.timesteps_at_ep)

    sm, off = moving_avg(cb.episode_ars, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_ars, color="seagreen", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off:off+len(sm)], sm, color="seagreen", linewidth=2,
             label=f"MaskablePPO (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Training Metrics — {scenario_name}  ({C.TOTAL_STEPS:,} steps)",
                  fontsize=12)
    ax1.grid(alpha=0.3)

    zero_viol = np.zeros_like(ts, dtype=float)
    ax2.plot(ts, zero_viol, color="tomato", alpha=0.4, linewidth=1.5,
             label="Violation rate (always 0 with masking)")
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    sm_p, off_p = moving_avg(cb.episode_placed, C.SMOOTH_W)
    ax3.plot(ts, cb.episode_placed, color="royalblue", alpha=0.2, linewidth=0.8)
    ax3.plot(ts[off_p:off_p+len(sm_p)], sm_p, color="royalblue", linewidth=2,
             label="services placed/ep")
    ax3.axhline(C.M, color="red", linestyle="--", alpha=0.5, label=f"M={C.M}")
    ax3.set_ylabel("Services Placed", fontsize=11)
    ax3.set_xlabel("Training steps", fontsize=11)
    ax3.set_ylim(0, C.M + 1)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_comparison(ilp_ar, rand_res, ppo_res, ppo_train_viol_mean, outdir, scenario_name):
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels = ["ILP\n(Optimal)", "Random\n(masked)", "MaskablePPO\n(P4)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"P2(ILP) vs Random(masked) vs P4(MaskablePPO) - {scenario_name}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    bp = ax.boxplot(
        [rand_res["ars"], ppo_res["ars"]],
        positions=[2, 3], widths=0.5, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors[1:]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.axhline(ilp_ar, color=colors[0], linestyle="--", linewidth=2, alpha=0.9,
               label=f"ILP  AR={ilp_ar:.4f}")
    ax.plot(1, ilp_ar, marker="D", color=colors[0], markersize=10, zorder=5)

    for pos, data, color in zip([2, 3], [rand_res["ars"], ppo_res["ars"]], colors[1:]):
        mv = np.mean(data)
        ax.text(pos, mv + 0.02, f"mu={mv:.3f}", ha="center", fontsize=9,
                fontweight="bold", color="black")

    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax.set_title("AR Distribution (0 violations)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    vr_means = [0.0, 0.0, ppo_train_viol_mean]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75)
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f"{v:.2%}", ha="center", fontsize=10, fontweight="bold", color="black")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_title("Violation Rate (eval)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = axes[2]
    pl_means = [C.M, np.mean(rand_res["placed"]), np.mean(ppo_res["placed"])]
    pl_stds  = [0.0, np.std(rand_res["placed"]), np.std(ppo_res["placed"])]
    bars = ax3.bar(labels, pl_means, color=colors, alpha=0.75,
                   yerr=pl_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, pl_means):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                 f"{v:.1f}", ha="center", fontsize=10, fontweight="bold", color="black")
    ax3.axhline(C.M, color="gray", linestyle=":", alpha=0.4)
    ax3.set_ylim(0, C.M + 2)
    ax3.set_ylabel("Services Placed per Episode", fontsize=11)
    ax3.set_title("Placement Completeness", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
@timer_utils.timer
def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    if args.total_timesteps is not None:
        C.TOTAL_STEPS = int(args.total_timesteps)
        print(f"[override] TOTAL_STEPS={C.TOTAL_STEPS:,}")
    print(f"\n{'='*60}")
    print(f"  P4 run_all.py  —  RL WITH action masking (0 violations)")
    print(f"  Config : {C.YAML_CONFIG.name}  |  scenario pool size={len(C.SCENARIOS)}  |  prototype idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    # 1. Load scenario
    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    # 2. ILP (all scenarios)
    print(f"\n[1/4] Solving ILP for all {len(C.SCENARIOS)} scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.SCENARIOS)} scenarios: {ilp_ar:.4f}")

    # 3. Random (masked)
    print(f"\n[2/4] Random masked baseline ({C.EVAL_EPS} episodes) ...")
    np.random.seed(C.SEED)
    rand_res = run_episodes(
        ecus, services,
        policy_fn=lambda obs, mask: int(np.random.choice(np.where(mask)[0]))
                                    if np.any(mask) else 0,
        n_eps=C.EVAL_EPS,
    )
    print(f"  Random AR  mean={np.mean(rand_res['ars']):.4f}  "
          f"std={np.std(rand_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(rand_res['placed']):.1f}/{M}")

    # 4. MaskablePPO training
    print(f"\n[3/4] MaskablePPO training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_maskppo(ecus, services, device)
    model.save(str(C.MODEL_PATH))
    print(f"  Model saved -> {C.MODEL_PATH}.zip")

    # 5. MaskablePPO evaluation
    print(f"\n[4/4] MaskablePPO evaluation ({C.EVAL_EPS} episodes, deterministic) ...")
    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return int(action)
    ppo_res = run_episodes(ecus, services, ppo_policy, C.EVAL_EPS)
    print(f"  PPO AR  mean={np.mean(ppo_res['ars']):.4f}  "
          f"std={np.std(ppo_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(ppo_res['placed']):.1f}/{M}")

    # Summary
    print(f"\n{'='*66}")
    print(f"  {'Method':<28} {'AR (mean+/-std)':<24} {'Placed':<10} {'Viol'}")
    print(f"  {'-'*28} {'-'*24} {'-'*10} {'-'*5}")
    print(f"  {'ILP (Optimal)':<28} {ilp_ar:.4f} +/- 0.0000       "
          f"{M}/{M:<7} 0")
    print(f"  {'Random (masked)':<28} "
          f"{np.mean(rand_res['ars']):.4f} +/- {np.std(rand_res['ars']):.4f}   "
          f"  {np.mean(rand_res['placed']):.1f}/{M:<4}  0")
    print(f"  {'MaskablePPO (P4)':<28} "
          f"{np.mean(ppo_res['ars']):.4f} +/- {np.std(ppo_res['ars']):.4f}   "
            f"  {np.mean(ppo_res['placed']):.1f}/{M:<4}  0")
    print(f"{'='*66}\n")

    # Save JSON
    log = {
        "scenario": sc_name,
        "prototype_scenario": prototype_name,
        "scenario_count": len(C.SCENARIOS),
        "N": N,
        "M": M,
        "ilp": {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "random_masked": {
            "ar_mean":            round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":             round(float(np.std(rand_res["ars"])), 6),
            "placed_mean":        round(float(np.mean(rand_res["placed"])), 2),
            "violations":         0,
            "cap_viol_total":     0,
            "conflict_viol_total": 0,
        },
        "maskable_ppo": {
            "ar_mean":            round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":             round(float(np.std(ppo_res["ars"])), 6),
            "placed_mean":        round(float(np.mean(ppo_res["placed"])), 2),
            "violations":         0,
            "cap_viol_total":     0,
            "conflict_viol_total": 0,
        },
        "training": {
            "total_steps": C.TOTAL_STEPS,
            "n_episodes":  len(cb.episode_ars),
            "ar_last50":   round(float(np.mean(cb.episode_ars[-50:])), 6),
            "viol_rate_last50": 0.0,
        },
    }

    base_path = C.OUTDIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_path.mkdir(parents=True, exist_ok=True)
    log_path = base_path / "results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {log_path}")

    # Save CSV summary
    csv_path = base_path / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "viol_rate", "cap_viol_total", "conflict_viol_total"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, 0.0, 0, 0])
        writer.writerow([
            "Random (masked)",
            round(float(np.mean(rand_res["ars"])), 6),
            round(float(np.std(rand_res["ars"])), 6),
            round(float(np.mean(rand_res["placed"])), 2),
            0.0, 0, 0,
        ])
        writer.writerow([
            "MaskablePPO (P4)",
            round(float(np.mean(ppo_res["ars"])), 6),
            round(float(np.std(ppo_res["ars"])), 6),
            round(float(np.mean(ppo_res["placed"])), 2),
            0.0, 0, 0,
        ])
    print(f"  CSV  saved -> {csv_path}")

    # Plots
    plot_training_curve(cb, ilp_ar, base_path, sc_name)
    plot_comparison(ilp_ar, rand_res, ppo_res, 0.0, base_path, sc_name)

    print("\nAll done! Output files:")
    print(f"  {base_path}/training_curve.png")
    print(f"  {base_path}/comparison.png")
    print(f"  {base_path}/results.json")
    print(f"  {base_path}/summary.csv\n")


if __name__ == "__main__":
    main()
