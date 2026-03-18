"""
train.py — Train a PPO agent with Lagrangian constraint relaxation (P5).

Dual-ascent λ update (every LAMBDA_UPDATE_WINDOW episodes):
    λ ← clip(λ + LAMBDA_LR * (avg_viol_rate − LAMBDA_TARGET), 0, LAMBDA_MAX)

Training dynamics:
  Early (λ ≈ 0): agent maximises AR freely; many violations.
  Mid   (λ ↑)  : violations become increasingly costly; agent learns to avoid them.
  End   (λ stable): agent balances AR and feasibility at equilibrium.

Run:
    python problem5_lagarange/train.py
"""

import datetime
import os
import functools
import sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from collections import deque

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import random
import config as C
from problem5_lagrangian.env import LagrangeEnv
from problem2_single.objects import ECU, SVC


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(cfg: str) -> str:
    if cfg != "auto":
        return cfg
    if torch.cuda.is_available():
        print(f"[CUDA] {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[CPU] No CUDA GPU, using CPU")
    return "cpu"


# 模块级工厂函数（必须可 pickle，SubprocVecEnv 需要在子进程中 spawn）
def _make_train_env(seed: int) -> Monitor:
    random.seed(seed)
    caps, reqs = C.SCENARIOS[C.SCENARIO_IDX]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env = LagrangeEnv(ecus, services, scenarios=C.SCENARIOS,
                      lambda_init=C.LAMBDA_INIT)
    return Monitor(env)


def moving_avg(arr, w: int):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    return np.convolve(arr, np.ones(w) / w, mode="valid"), w - 1


# ─────────────────────────────────────────────────────────────────────────────
#  Lagrangian Callback (dual ascent)
# ─────────────────────────────────────────────────────────────────────────────

class LagrangeCallback(BaseCallback):
    """
    After every LAMBDA_UPDATE_WINDOW completed episodes, update λ:

        λ ← clip(λ + LAMBDA_LR * (mean_viol_rate − LAMBDA_TARGET), 0, LAMBDA_MAX)

    The new λ is propagated to all envs via monitor_env.env.set_lambda(λ).
    (DummyVecEnv: training_env.envs[i] is Monitor(LagrangeEnv),
     so training_env.envs[i].env is LagrangeEnv.)
    """

    def __init__(self):
        super().__init__()
        self.lambda_val          = C.LAMBDA_INIT
        self._viol_window        = deque(maxlen=C.LAMBDA_UPDATE_WINDOW)
        # Per-episode logs
        self.episode_ars:        list[float] = []
        self.episode_viol_rates: list[float] = []
        self.episode_lambdas:    list[float] = []
        self.timesteps_at_ep:    list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            # True AR from LagrangeEnv's last-step info (not Monitor's penalised total)
            ar        = float(info.get("ar", 0.0))
            viol_rate = float(info.get("viol_rate_ep", 0.0))

            self.episode_ars.append(ar)
            self.episode_viol_rates.append(viol_rate)
            self.episode_lambdas.append(self.lambda_val)
            self.timesteps_at_ep.append(self.num_timesteps)

            # Dual ascent: update λ every LAMBDA_UPDATE_WINDOW episodes
            self._viol_window.append(viol_rate)
            if len(self._viol_window) == C.LAMBDA_UPDATE_WINDOW:
                avg_viol        = float(np.mean(self._viol_window))
                new_lam         = self.lambda_val + C.LAMBDA_LR * (avg_viol - C.LAMBDA_TARGET)
                self.lambda_val = float(np.clip(new_lam, 0.0, C.LAMBDA_MAX))
                # 兼容 DummyVecEnv 和 SubprocVecEnv
                self.training_env.env_method("set_lambda", self.lambda_val)

        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Build model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(env, device: str) -> PPO:
    return PPO(
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


# ─────────────────────────────────────────────────────────────────────────────
#  Plot training curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(cb: LagrangeCallback, outdir: Path):
    ts  = np.array(cb.timesteps_at_ep)
    ars = np.array(cb.episode_ars)
    vrs = np.array(cb.episode_viol_rates)
    lms = np.array(cb.episode_lambdas)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── AR ──
    sm, off = moving_avg(ars, C.SMOOTH_W)
    ax1.plot(ts, ars, color="seagreen", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off:off+len(sm)], sm, color="seagreen", linewidth=2,
             label=f"AR (smoothed w={C.SMOOTH_W})")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_title(
        f"P5 Lagrangian PPO Training — Scenario {C.SCENARIO_IDX+1}"
        f"  ({C.TOTAL_STEPS:,} steps)",
        fontsize=12,
    )
    ax1.grid(alpha=0.3)

    # ── Violation rate (left axis) + λ (right axis) ──
    sm_v, off_v = moving_avg(vrs, C.SMOOTH_W)
    ax2.plot(ts, vrs, color="tomato", alpha=0.2, linewidth=0.8)
    ax2.plot(ts[off_v:off_v+len(sm_v)], sm_v, color="tomato", linewidth=2,
             label="Viol rate (smoothed)")
    ax2.set_ylabel("Violation Rate / Episode", fontsize=11, color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax2.set_ylim(-0.05, 1.1)

    ax2b = ax2.twinx()
    ax2b.plot(ts, lms, color="navy", linewidth=1.5, linestyle="--", alpha=0.85,
              label="λ value")
    ax2b.set_ylabel("λ (Lagrangian multiplier)", fontsize=11, color="navy")
    ax2b.tick_params(axis="y", labelcolor="navy")
    ax2b.set_ylim(bottom=0)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(C.DEVICE)

    print(f"\n{'='*60}")
    print(f"  Problem 5 — Lagrangian PPO")
    print(f"  Constraints: soft, penalised by λ (no action masking).")
    n_envs = os.cpu_count() or 1
    print(f"  N={C.N}  M={C.M}  steps={C.TOTAL_STEPS:,}  device={device.upper()}")
    print(f"  λ_init={C.LAMBDA_INIT}  lr={C.LAMBDA_LR}  target={C.LAMBDA_TARGET}  max={C.LAMBDA_MAX}")
    print(f"  CPU cores detected: {os.cpu_count()}  →  parallel envs: {n_envs}")
    print(f"{'='*60}\n")

    env_fns = [functools.partial(_make_train_env, C.SEED + i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    cb  = LagrangeCallback()
    model = build_model(env, device)

    print(f"Training ({n_envs} parallel envs) ...")
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0

    n_ep        = len(cb.episode_ars)
    last50_ar   = np.mean(cb.episode_ars[-50:])        if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_viol = np.mean(cb.episode_viol_rates[-50:]) if n_ep >= 50 else np.mean(cb.episode_viol_rates)

    print(f"\n  Elapsed      : {elapsed:.1f}s  |  episodes : {n_ep}")
    print(f"  AR  (last 50): {last50_ar:.4f}")
    print(f"  Viol (last 50): {last50_viol:.4f}")
    print(f"  Final λ       : {cb.lambda_val:.4f}")

    model.save(str(C.MODEL_PATH))
    print(f"  Model saved -> {C.MODEL_PATH}.zip")

    log = {
        "N": C.N, "M": C.M,
        "total_steps": C.TOTAL_STEPS,
        "device": device,
        "elapsed_s": round(elapsed, 2),
        "n_episodes": n_ep,
        "ar_last50":        round(float(last50_ar), 6),
        "viol_rate_last50": round(float(last50_viol), 6),
        "final_lambda":     round(float(cb.lambda_val), 6),
        "episode_ars":        cb.episode_ars,
        "episode_viol_rates": cb.episode_viol_rates,
        "episode_lambdas":    cb.episode_lambdas,
        "timesteps":          cb.timesteps_at_ep,
    }
    run_dir = C.OUTDIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved  -> {run_dir / 'training_log.json'}")

    plot_training_curve(cb, run_dir)
    env.close()
    print("\nDone. Run evaluate.py or run.py for full comparison.\n")


if __name__ == "__main__":
    main()
