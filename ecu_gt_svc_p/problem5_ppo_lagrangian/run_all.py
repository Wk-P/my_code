"""
run_all.py — One-shot P5 full pipeline (Lagrangian Constraint Relaxation):

  1. Load scenario from YAML
  2. Solve with ILP (PuLP)              -> ilp_ar (optimal upper bound)
  3. Evaluate random policy (no mask)   -> random results
  4. Train Lagrangian PPO               -> training curve  (AR + viol rate + λ)
  5. Evaluate trained Lagrangian PPO    -> ppo results
  6. Produce plots:
       training_curve.png  — AR, violation rate, λ evolution during training
       comparison.png      — 3-way: ILP vs Random vs Lagrange PPO  (AR + viol rate)
       results.json        — full numeric summary

P5 Design: constraints are SOFT, penalised by adaptive λ. No action masking.
    - Episode always runs M steps (no early termination for violations).
    - Reward per step: n_i / e_j - λ * c_t
    - λ is included in the observation to reduce non-stationarity.
    - Dual ascent: λ ← clip(λ + lr*(avg_viol - target), 0, λ_max)

Run:
    python problem5_ppo_lagrangian/run_all.py
"""

import datetime
import csv
import os
import argparse
import functools
import sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import timer_utils

import torch
import yaml
import pulp
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import random
import config as C
from problem5_ppo_lagrangian.env import LagrangeEnv
from problem2_ilp.objects import ECU, SVC
from run_utils import parse_args, resolve_device, moving_avg, solve_ilp, solve_ilp_all_scenarios, load_scenario


# ─────────────────────────────────────────────────────────────────────────────
#  DummyVecEnv factory (module-level → picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _make_lagrange_env(seed: int) -> Monitor:
    random.seed(seed)
    caps, reqs, _ = C.SCENARIOS[C.SCENARIO_IDX]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env = LagrangeEnv(ecus, services, scenarios=C.SCENARIOS,
                      lambda_init=C.LAMBDA_INIT,
                      lambda_max=C.LAMBDA_MAX)
    return Monitor(env)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 & 5 — Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps, lambda_eval: float = 0.0):
    """policy_fn(obs) -> int. Evaluation uses a fixed λ value in the observation."""
    env = LagrangeEnv(ecus, services, scenarios=C.SCENARIOS,
                      lambda_init=lambda_eval, lambda_max=C.LAMBDA_MAX)
    ars, viol_rates, viols, placed_list = [], [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            obs, _, done, _, info = env.step(policy_fn(obs))
        ars.append(info.get("ar", 0.0))
        viol_rates.append(info.get("viol_rate_ep", 0.0))
        viols.append(int(info.get("violations_ep", 0)))
        placed_list.append(info.get("services_placed", 0))
    return {
        "ars":        np.array(ars),
        "viol_rates": np.array(viol_rates),
        "viols":      np.array(viols),
        "placed":     np.array(placed_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Lagrangian PPO training
# ══════════════════════════════════════════════════════════════════════════════

class LagrangeCallback(BaseCallback):
    """
    Dual-ascent λ update every LAMBDA_UPDATE_WINDOW episodes.
    Propagates updated λ to all envs in DummyVecEnv.
    """
    def __init__(self):
        super().__init__()
        self.lambda_val          = C.LAMBDA_INIT
        self._viol_window        = deque(maxlen=C.LAMBDA_UPDATE_WINDOW)
        self._episode_count      = 0
        self._episodes_since_lambda_update = 0
        self.episode_ars:        list[float] = []
        self.episode_viol_rates: list[float] = []
        self.episode_placed:     list[int]   = []
        self.episode_lambdas:    list[float] = []
        self.timesteps_at_ep:    list[int]   = []
        self._next_progress_step = C.PROGRESS_LOG_EVERY_STEPS
        self._t_start = 0.0

    def _on_training_start(self) -> None:
        self._t_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self._episode_count += 1
            ar        = float(info.get("ar", 0.0))
            viol_rate = float(info.get("viol_rate_ep", 0.0))
            self.episode_ars.append(ar)
            self.episode_viol_rates.append(viol_rate)
            self.episode_placed.append(int(info.get("services_placed", 0)))
            self.episode_lambdas.append(self.lambda_val)
            self.timesteps_at_ep.append(self.num_timesteps)

            self._viol_window.append(viol_rate)
            if self._episode_count >= C.LAMBDA_WARMUP_EPISODES:
                self._episodes_since_lambda_update += 1

            if (
                self._episode_count >= C.LAMBDA_WARMUP_EPISODES
                and len(self._viol_window) == C.LAMBDA_UPDATE_WINDOW
                and self._episodes_since_lambda_update >= C.LAMBDA_UPDATE_WINDOW
            ):
                avg_viol        = float(np.mean(self._viol_window))
                new_lam         = self.lambda_val + C.LAMBDA_LR * (avg_viol - C.LAMBDA_TARGET)
                self.lambda_val = float(np.clip(new_lam, 0.0, C.LAMBDA_MAX))
                self.training_env.env_method("set_lambda", self.lambda_val)
                self._episodes_since_lambda_update = 0

        if self.num_timesteps >= self._next_progress_step:
            elapsed = max(time.time() - self._t_start, 1e-6)
            pct = min(100.0, self.num_timesteps * 100.0 / C.TOTAL_STEPS)
            eps = len(self.episode_ars)
            sps = self.num_timesteps / elapsed
            print(
                f"  [train] step={self.num_timesteps:,}/{C.TOTAL_STEPS:,} "
                f"({pct:5.1f}%) | eps={eps} | steps/s={sps:,.0f} | lambda={self.lambda_val:.4f}"
            )
            self._next_progress_step += C.PROGRESS_LOG_EVERY_STEPS
        return True


def train_lagrange(device: str, n_envs: int = 1):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    env_fns = [functools.partial(_make_lagrange_env, C.SEED + i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns, start_method=C.SUBPROC_START_METHOD)
    print(f"  Using SubprocVecEnv: n_envs={n_envs}, start_method={C.SUBPROC_START_METHOD}")
    cb  = LagrangeCallback()

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
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb, progress_bar=False)
    elapsed = time.time() - t0
    env.close()

    n_ep        = len(cb.episode_ars)
    last50_ar   = np.mean(cb.episode_ars[-50:])        if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_viol = np.mean(cb.episode_viol_rates[-50:]) if n_ep >= 50 else np.mean(cb.episode_viol_rates)
    print(f"  Training done  {elapsed:.1f}s | {n_ep} episodes"
          f" | AR(last50)={last50_ar:.4f}"
          f" | viol(last50)={last50_viol:.4f}"
          f" | final λ={cb.lambda_val:.4f}")
    return model, cb


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curve(cb: LagrangeCallback, ilp_ar: float,
                        outdir: Path, scenario_name: str):
    ts  = np.array(cb.timesteps_at_ep)
    ars = np.array(cb.episode_ars)
    vrs = np.array(cb.episode_viol_rates)
    pls = np.array(cb.episode_placed)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # ── AR ──
    sm, off = moving_avg(ars, C.SMOOTH_W)
    ax1.plot(ts, ars, color="seagreen", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off:off+len(sm)], sm, color="seagreen", linewidth=2,
             label=f"AR (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_title(
        f"Training Metrics — {scenario_name}  ({C.TOTAL_STEPS:,} steps)",
        fontsize=12,
    )
    ax1.grid(alpha=0.3)

    sm_v, off_v = moving_avg(vrs, C.SMOOTH_W)
    ax2.plot(ts, vrs, color="tomato", alpha=0.2, linewidth=0.8)
    ax2.plot(ts[off_v:off_v+len(sm_v)], sm_v, color="tomato", linewidth=2,
             label="Viol rate (smoothed)")
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)

    sm_p, off_p = moving_avg(pls, C.SMOOTH_W)
    ax3.plot(ts, pls, color="royalblue", alpha=0.2, linewidth=0.8)
    ax3.plot(ts[off_p:off_p+len(sm_p)], sm_p, color="royalblue", linewidth=2,
             label="Services placed (smoothed)")
    ax3.axhline(C.M, color="gray", linestyle=":", alpha=0.6, label=f"M={C.M}")
    ax3.set_ylabel("Services Placed", fontsize=11)
    ax3.set_xlabel("Training steps", fontsize=11)
    ax3.set_ylim(0, C.M + 1)
    ax3.legend(fontsize=9, loc="lower right")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_comparison(ilp_ar, rand_res, ppo_res, ppo_train_viol_mean, ppo_train_viol_std, outdir: Path, scenario_name: str):
    colors = ["#e74c3c", "#3498db", "#e67e22"]
    labels = ["ILP\n(Optimal)", "Random\n(no mask)", "Lagrange\nPPO"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"P2(ILP) vs Random vs P5(Lagrange PPO) - {scenario_name}",
        fontsize=13, fontweight="bold",
    )

    # ── AR ──
    bp = ax1.boxplot(
        [rand_res["ars"], ppo_res["ars"]],
        positions=[2, 3], widths=0.5, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors[1:]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax1.axhline(ilp_ar, color=colors[0], linestyle="--", linewidth=2, alpha=0.9,
                label=f"ILP  AR={ilp_ar:.4f}")
    ax1.plot(1, ilp_ar, marker="D", color=colors[0], markersize=10, zorder=5)

    for pos, data, color in zip([2, 3], [rand_res["ars"], ppo_res["ars"]], colors[1:]):
        mv = np.mean(data)
        ax1.text(pos, mv + 0.02, f"mu={mv:.3f}", ha="center", fontsize=9,
                 fontweight="bold", color="black")

    ax1.set_xticks([1, 2, 3]); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax1.set_title("AR Distribution", fontsize=11)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(axis="y", alpha=0.3)

    vr_means = [0.0, np.mean(rand_res["viol_rates"]), ppo_train_viol_mean]
    vr_stds  = [0.0, np.std(rand_res["viol_rates"]),  ppo_train_viol_std]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75,
                   yerr=vr_stds, capsize=6, ecolor="black")
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.01,
                 f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color="black")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_title("Violation Rate (eval)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    pl_means = [C.M, np.mean(rand_res["placed"]), np.mean(ppo_res["placed"])]
    pl_stds  = [0.0, np.std(rand_res["placed"]), np.std(ppo_res["placed"])]
    bars = ax3.bar(labels, pl_means, color=colors, alpha=0.75,
                   yerr=pl_stds, capsize=6, ecolor="black")
    for bar, v in zip(bars, pl_means):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.05,
                 f"{v:.1f}", ha="center", fontsize=10, fontweight="bold", color="black")
    ax3.set_ylim(0, C.M + 1)
    ax3.set_ylabel("Services Placed", fontsize=11)
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
    device = resolve_device(C.DEVICE)

    print(f"\n{'='*60}")
    print(f"  P5 run.py  —  Lagrangian Constraint Relaxation")
    n_envs = max(1, int(C.N_ENVS))
    print(f"  Config : {C.YAML_CONFIG.name}  |  scenario pool size={len(C.SCENARIOS)}  |  prototype idx={C.SCENARIO_IDX}")
    print(f"  λ_init={C.LAMBDA_INIT}  lr={C.LAMBDA_LR}  target={C.LAMBDA_TARGET}  max={C.LAMBDA_MAX}")
    print(f"  CPU cores detected: {os.cpu_count()}  →  configured parallel envs: {n_envs}")
    print(f"{'='*60}\n")

    # 1. Load scenario
    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    # 2. ILP (all scenarios)
    print(f"\n[1/4] Solving ILP for all {len(C.SCENARIOS)} scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.SCENARIOS)} scenarios: {ilp_ar:.4f}")

    # 3. Random baseline (no masking)
    print(f"\n[2/4] Random baseline ({C.EVAL_EPS} episodes, no masking) ...")
    np.random.seed(C.SEED)
    rand_res = run_episodes(
        ecus, services,
        policy_fn=lambda obs: int(np.random.randint(0, N)),
        n_eps=C.EVAL_EPS,
        lambda_eval=0.0,
    )
    print(f"  Random  AR mean={np.mean(rand_res['ars']):.4f}  "
          f"viol={np.mean(rand_res['viol_rates']):.2%}")

    # 4. Lagrangian PPO training
    print(f"\n[3/4] Lagrangian PPO training ({C.TOTAL_STEPS:,} steps, {n_envs} envs) ...")
    model, cb = train_lagrange(device, n_envs)
    model.save(str(C.MODEL_PATH))
    print(f"  Model saved -> {C.MODEL_PATH}.zip")

    # 5. Lagrangian PPO evaluation
    print(f"\n[4/4] Lagrangian PPO evaluation ({C.EVAL_EPS} episodes, deterministic) ...")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ppo_res = run_episodes(ecus, services, ppo_policy, C.EVAL_EPS, lambda_eval=cb.lambda_val)
    print(f"  PPO  AR mean={np.mean(ppo_res['ars']):.4f}  "
          f"Eval viol={np.mean(ppo_res['viol_rates']):.2%}")

    # Summary table
    rand_viol_str = f"{np.mean(rand_res['viol_rates']):.2%}"
    ppo_train_viol = float(np.mean(ppo_res["viol_rates"]))
    ppo_train_viol_std = float(np.std(ppo_res["viol_rates"]))
    ppo_viol_str  = f"{ppo_train_viol:.2%}"
    print(f"\n{'='*72}")
    print(f"  {'Method':<28} {'AR mean±std':<24} {'Viol%':<14} {'Placed'}")
    print(f"  {'-'*28} {'-'*24} {'-'*14} {'-'*8}")
    print(f"  {'ILP (Optimal)':<28} {ilp_ar:.4f} ± 0.0000        {'0.00%':<14} {M}/{M}")
    print(f"  {'Random (no mask)':<28} "
          f"{np.mean(rand_res['ars']):.4f} ± {np.std(rand_res['ars']):.4f}   "
          f"  {rand_viol_str:<12} {M}/{M}")
    print(f"  {'Lagrange PPO':<28} "
          f"{np.mean(ppo_res['ars']):.4f} ± {np.std(ppo_res['ars']):.4f}   "
          f"  {ppo_viol_str:<12} {M}/{M}")
    print(f"  Final λ = {cb.lambda_val:.4f}")
    print(f"{'='*72}\n")

    # Save JSON results
    n_ep = len(cb.episode_ars)
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
        "random": {
            "ar_mean":        round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":         round(float(np.std(rand_res["ars"])), 6),
            "viol_rate_mean": round(float(np.mean(rand_res["viol_rates"])), 6),
        },
        "lagrange_ppo": {
            "ar_mean":        round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":         round(float(np.std(ppo_res["ars"])), 6),
            "viol_rate_mean": round(float(ppo_train_viol), 6),
        },
        "training": {
            "total_steps":    C.TOTAL_STEPS,
            "n_episodes":     n_ep,
            "ar_last50":      round(float(np.mean(cb.episode_ars[-50:])),        6) if n_ep >= 50 else None,
            "viol_last50":    round(float(np.mean(cb.episode_viol_rates[-50:])), 6) if n_ep >= 50 else None,
            "final_lambda":   round(float(cb.lambda_val), 6),
            "eval_lambda":    round(float(cb.lambda_val), 6),
        },
    }
    run_dir = C.OUTDIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {run_dir / 'results.json'}")

    # Save CSV summary
    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "viol_rate"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, 0.0])
        writer.writerow([
            "Random (no mask)",
            round(float(np.mean(rand_res["ars"])), 6),
            round(float(np.std(rand_res["ars"])), 6),
            round(float(np.mean(rand_res["placed"])), 2),
            round(float(np.mean(rand_res["viol_rates"])), 4),
        ])
        writer.writerow([
            "Lagrange PPO",
            round(float(np.mean(ppo_res["ars"])), 6),
            round(float(np.std(ppo_res["ars"])), 6),
            round(float(np.mean(ppo_res["placed"])), 2),
            round(float(ppo_train_viol), 4),
        ])
    print(f"  CSV  saved -> {csv_path}")

    # Plots
    plot_training_curve(cb, ilp_ar, run_dir, sc_name)
    plot_comparison(ilp_ar, rand_res, ppo_res, ppo_train_viol, ppo_train_viol_std, run_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {run_dir}/training_curve.png")
    print(f"  {run_dir}/comparison.png")
    print(f"  {run_dir}/results.json")
    print(f"  {run_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
