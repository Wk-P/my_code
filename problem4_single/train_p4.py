"""
train_p4.py — Train a MaskablePPO agent on the P4 environment (WITH action masking).

Action masking guarantees 0 constraint violations by construction.
The agent can only choose ECUs that have enough capacity AND haven't been used.

Run:
    python problem4_single/train_p4.py

Outputs saved to problem4_single/results/:
    maskppo_p4_model.zip  — trained MaskablePPO weights
    training_curve.png    — per-episode AR over time
    training_log.json     — raw episode data
"""

import sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import random
import config as C
from env_p4 import P4Env
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


def mask_fn(env) -> np.ndarray:
    """Callable that MaskablePPO uses to get the current action mask."""
    return env.action_masks()


def make_env(seed: int = 0) -> Monitor:
    random.seed(seed)
    caps, reqs = C.SCENARIOS[0]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env = P4Env(ecus, services, scenarios=C.SCENARIOS)
    env = ActionMasker(env, mask_fn)
    return Monitor(env)


def moving_avg(arr, w: int):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    return np.convolve(arr, np.ones(w) / w, mode="valid"), w - 1


# ─────────────────────────────────────────────────────────────────────────────
#  Callback
# ─────────────────────────────────────────────────────────────────────────────

class P4Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_ars:     list[float] = []
        self.episode_placed:  list[int]   = []
        self.timesteps_at_ep: list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ar = float(info["episode"]["r"])
                placed = int(info.get("services_placed", 0))
                self.episode_ars.append(ar)
                self.episode_placed.append(placed)
                self.timesteps_at_ep.append(self.num_timesteps)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Build model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(env: Monitor, device: str) -> MaskablePPO:
    return MaskablePPO(
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

def plot_training_curve(cb: P4Callback, outdir: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ts = np.array(cb.timesteps_at_ep)

    sm, off = moving_avg(cb.episode_ars, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_ars, color="seagreen", alpha=0.25, linewidth=0.8, label="raw AR")
    ax1.plot(ts[off:off+len(sm)], sm, color="seagreen", linewidth=2,
             label=f"smoothed (w={C.SMOOTH_W})")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.set_title(f"P4 MaskablePPO Training  |  N={C.N}  M={C.M}  steps={C.TOTAL_STEPS:,}", fontsize=12)
    ax1.grid(alpha=0.3)

    sm_p, off_p = moving_avg(cb.episode_placed, C.SMOOTH_W)
    ax2.plot(ts, cb.episode_placed, color="royalblue", alpha=0.25, linewidth=0.8, label="raw placed")
    ax2.plot(ts[off_p:off_p+len(sm_p)], sm_p, color="royalblue", linewidth=2,
             label=f"smoothed (w={C.SMOOTH_W})")
    ax2.set_ylabel("Services Placed / Episode", fontsize=11)
    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.set_ylim(0, C.M + 1)
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
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(C.DEVICE)

    print(f"\n{'='*60}")
    print(f"  Problem 4 — MaskablePPO (WITH action masking)")
    print(f"  Constraint violations = 0 by construction.")
    print(f"  N={C.N}  M={C.M}  steps={C.TOTAL_STEPS:,}  device={device.upper()}")
    print(f"{'='*60}\n")

    env = make_env(seed=C.SEED)
    cb  = P4Callback()
    model = build_model(env, device)

    print("Training ...")
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0

    n_ep   = len(cb.episode_ars)
    last50 = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_p = np.mean(cb.episode_placed[-50:]) if n_ep >= 50 else np.mean(cb.episode_placed)

    print(f"\n  Elapsed : {elapsed:.1f}s  |  episodes : {n_ep}")
    print(f"  AR  (last 50 eps)         : {last50:.4f}")
    print(f"  Placed (last 50 eps)      : {last50_p:.1f}/{C.M}")

    model.save(str(C.MODEL_PATH))
    print(f"  Model saved -> {C.MODEL_PATH}.zip")

    log = {
        "N": C.N, "M": C.M,
        "total_steps": C.TOTAL_STEPS,
        "device": device,
        "elapsed_s": round(elapsed, 2),
        "n_episodes": n_ep,
        "ar_last50": round(float(last50), 6),
        "placed_last50": round(float(last50_p), 2),
        "episode_ars":    cb.episode_ars,
        "episode_placed": cb.episode_placed,
        "timesteps":      cb.timesteps_at_ep,
    }
    log_path = C.OUTDIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved  -> {log_path}")

    plot_training_curve(cb, C.OUTDIR)

    env.close()
    print("\nDone. Run evaluate_p4.py or run_all.py for full comparison.\n")


if __name__ == "__main__":
    main()
