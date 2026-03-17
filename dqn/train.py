"""
train.py — Train a DQN agent on the DQN environment (NO action masking).

Constraint violations terminate the episode with reward = -1.
The agent learns to avoid violations through reward shaping.

Run:
    python dqn/train.py

Outputs saved to dqn/results/<timestamp>/:
    dqn_model.zip       — trained DQN weights
    training_curve.png  — per-episode reward and violation rate over time
    training_log.json   — raw episode data
"""

import datetime
import sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import random
import config as C
from dqn.env import DQNEnv
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


def make_env(seed: int = 0) -> Monitor:
    random.seed(seed)
    caps, reqs = C.SCENARIOS[C.SCENARIO_IDX]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env = DQNEnv(ecus, services, scenarios=C.SCENARIOS)
    return Monitor(env)

def moving_avg(arr, w: int):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    return np.convolve(arr, np.ones(w) / w, mode="valid"), w - 1


# ─────────────────────────────────────────────────────────────────────────────
#  Callback
# ─────────────────────────────────────────────────────────────────────────────

class DQNCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards:  list[float] = []
        self.episode_placed:   list[int]   = []
        self.episode_violated: list[int]   = []
        self.timesteps_at_ep:  list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_placed.append(int(info.get("services_placed", 0)))
                self.episode_violated.append(1 if info.get("violated", False) else 0)
                self.timesteps_at_ep.append(self.num_timesteps)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Build model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(env: Monitor, device: str) -> DQN:
    return DQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = C.DQN_LR,
        buffer_size            = C.DQN_BUFFER_SIZE,
        learning_starts        = C.DQN_LEARNING_STARTS,
        batch_size             = C.DQN_BATCH_SIZE,
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


# ─────────────────────────────────────────────────────────────────────────────
#  Plot training curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(cb: DQNCallback, outdir: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ts = np.array(cb.timesteps_at_ep)

    sm, off = moving_avg(cb.episode_rewards, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_rewards, color="seagreen", alpha=0.25, linewidth=0.8, label="raw reward")
    ax1.plot(ts[off:off+len(sm)], sm, color="seagreen", linewidth=2,
             label=f"smoothed (w={C.SMOOTH_W})")
    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_title(f"DQN Training  |  N={C.N}  M={C.M}  steps={C.TOTAL_STEPS:,}", fontsize=12)
    ax1.grid(alpha=0.3)

    viol_rate = np.array(cb.episode_violated, dtype=float)
    sm_v, off_v = moving_avg(viol_rate, C.SMOOTH_W)
    ax2.plot(ts, viol_rate, color="tomato", alpha=0.25, linewidth=0.8)
    ax2.plot(ts[off_v:off_v+len(sm_v)], sm_v, color="tomato", linewidth=2,
             label=f"violation rate (smoothed)")
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(fontsize=9)
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
    run_dir = C.OUTDIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(C.DEVICE)

    print(f"\n{'='*60}")
    print(f"  DQN  (NO action masking)")
    print(f"  Violation \u2192 reward=-1, episode terminates.")
    print(f"  N={C.N}  M={C.M}  steps={C.TOTAL_STEPS:,}  device={device.upper()}")
    print(f"{'='*60}\n")

    env   = make_env(seed=C.SEED)
    cb    = DQNCallback()
    model = build_model(env, device)

    print("Training ...")
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0

    n_ep        = len(cb.episode_rewards)
    last50_r    = np.mean(cb.episode_rewards[-50:]) if n_ep >= 50 else np.mean(cb.episode_rewards)
    last50_p    = np.mean(cb.episode_placed[-50:])  if n_ep >= 50 else np.mean(cb.episode_placed)
    last50_viol = np.mean(cb.episode_violated[-50:]) if n_ep >= 50 else np.mean(cb.episode_violated)

    print(f"\n  Elapsed : {elapsed:.1f}s  |  episodes : {n_ep}")
    print(f"  Reward (last 50 eps)      : {last50_r:.4f}")
    print(f"  Placed (last 50 eps)      : {last50_p:.1f}/{C.M}")
    print(f"  Violation rate (last 50)  : {last50_viol:.2%}")

    model_path = run_dir / "dqn_model"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    log = {
        "N": C.N, "M": C.M,
        "total_steps":      C.TOTAL_STEPS,
        "device":           device,
        "elapsed_s":        round(elapsed, 2),
        "n_episodes":       n_ep,
        "reward_last50":    round(float(last50_r), 6),
        "placed_last50":    round(float(last50_p), 2),
        "viol_rate_last50": round(float(last50_viol), 4),
        "episode_rewards":  cb.episode_rewards,
        "episode_placed":   cb.episode_placed,
        "episode_violated": cb.episode_violated,
        "timesteps":        cb.timesteps_at_ep,
    }
    log_path = run_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved  -> {log_path}")

    plot_training_curve(cb, run_dir)

    env.close()
    print(f"\nDone. Results in {run_dir}\n")


if __name__ == "__main__":
    main()
