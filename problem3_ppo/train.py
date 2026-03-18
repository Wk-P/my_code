"""
train_p3.py — Train a PPO agent on the P3 environment (NO constraint enforcement).

The P3 env never terminates early: violations are recorded but not penalized.
The only reward is the final AR at the last step.

Run:
    python problem3_ppo/train_p3.py

Outputs saved to problem3_ppo/results/:
    ppo_p3_model.zip      — trained PPO weights
    training_curve.png    — per-episode AR and violation count over time
    training_log.json     — raw episode data
"""

import datetime
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import random
import config as C
from problem3_ppo.env import P3Env
from problem2_ilp.objects import ECU, SVC


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(cfg: str) -> str:
    if cfg != "auto":
        return cfg
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[CUDA] {name}")
        return "cuda"
    print("[CPU] No CUDA GPU, using CPU")
    return "cpu"


def make_env(seed: int = 0) -> Monitor:
    random.seed(seed)
    caps, reqs = C.SCENARIOS[0]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(P3Env(ecus, services, scenarios=C.SCENARIOS))


def moving_avg(arr, w: int):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    sm = np.convolve(arr, np.ones(w) / w, mode="valid")
    return sm, w - 1


# ─────────────────────────────────────────────────────────────────────────────
#  Callback — records AR and violations each episode
# ─────────────────────────────────────────────────────────────────────────────

class P3Callback(BaseCallback):
    """
    Episode always runs to completion (M steps).
    At done=True, Monitor injects info["episode"]; we also read total_violations.
    """

    def __init__(self):
        super().__init__()
        self.episode_ars:        list[float] = []
        self.episode_violations: list[int]   = []
        self.timesteps_at_ep:    list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ar   = float(info["episode"]["r"])
                viol = int(info.get("total_violations", 0))
                self.episode_ars.append(ar)
                self.episode_violations.append(viol)
                self.timesteps_at_ep.append(self.num_timesteps)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Build PPO model
# ─────────────────────────────────────────────────────────────────────────────

def build_ppo(env: Monitor, device: str) -> PPO:
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
#  Save training curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(cb: P3Callback, outdir: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ts = np.array(cb.timesteps_at_ep)

    # ── AR curve ─────────────────────────────────────────────────────────────
    sm, off = moving_avg(cb.episode_ars, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_ars, color="steelblue", alpha=0.25, linewidth=0.8, label="raw AR")
    ax1.plot(ts[off: off + len(sm)], sm,
             color="steelblue", linewidth=2, label=f"smoothed (w={C.SMOOTH_W})")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.set_title(f"P3 PPO Training  |  N={C.N}  M={C.M}  steps={C.TOTAL_STEPS:,}", fontsize=12)
    ax1.grid(alpha=0.3)

    # ── Violation count curve ─────────────────────────────────────────────────
    sm_v, off_v = moving_avg(cb.episode_violations, C.SMOOTH_W)
    ax2.plot(ts, cb.episode_violations, color="tomato", alpha=0.25, linewidth=0.8, label="raw violations/ep")
    ax2.plot(ts[off_v: off_v + len(sm_v)], sm_v,
             color="tomato", linewidth=2, label=f"smoothed (w={C.SMOOTH_W})")
    ax2.set_ylabel("Constraint Violations / Episode", fontsize=11)
    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(C.DEVICE)

    print(f"\n{'='*60}")
    print(f"  Problem 3 — PPO (NO constraint enforcement)")
    print(f"  Violations are recorded but NOT penalized.")
    print(f"  N={C.N}  M={C.M}  steps={C.TOTAL_STEPS:,}  device={device.upper()}")
    print(f"{'='*60}\n")

    env = make_env(seed=C.SEED)
    cb  = P3Callback()
    model = build_ppo(env, device)

    print("Training …")
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0

    n_ep     = len(cb.episode_ars)
    last50   = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_v = np.mean(cb.episode_violations[-50:]) if n_ep >= 50 else np.mean(cb.episode_violations)

    print(f"\n  Elapsed : {elapsed:.1f}s  |  episodes : {n_ep}")
    print(f"  AR  (last 50 eps)         : {last50:.4f}")
    print(f"  Violations/ep (last 50)   : {last50_v:.2f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(str(C.MODEL_PATH))
    print(f"  Model saved → {C.MODEL_PATH}.zip")

    # ── Save training log ─────────────────────────────────────────────────────
    log = {
        "N": C.N, "M": C.M,
        "total_steps": C.TOTAL_STEPS,
        "device": device,
        "elapsed_s": round(elapsed, 2),
        "n_episodes": n_ep,
        "ar_last50": round(float(last50), 6),
        "violations_per_ep_last50": round(float(last50_v), 4),
        "episode_ars":        cb.episode_ars,
        "episode_violations": cb.episode_violations,
        "timesteps":          cb.timesteps_at_ep,
    }
    log_path = C.OUTDIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = log_path.parent
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved  → {log_path}")

    # ── Plot ────────────────────────────────────────────────────────────────────────────
    plot_training_curve(cb, run_dir)

    env.close()
    print("\nDone. Run evaluate_p3.py to compare against random baseline.\n")


if __name__ == "__main__":
    # main()
    pass
