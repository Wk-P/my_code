"""
train_p3.py — Train a PPO agent on the P3 environment (no constraint enforcement).

Run:
    python problem3_single/train_p3.py

Outputs saved to problem3_single/results/:
    ppo_p3_model.zip      — trained PPO weights
    training_curve.png    — per-episode AR and violation rate over time
    training_log.json     — raw episode data (AR, violations, timestep)
"""

import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# ── resolve imports ───────────────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import random
import config as C
from env_p3 import P3Env
from problem2_single.objects import ECU, SVC


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(cfg: str) -> str:
    if cfg != "auto":
        return cfg
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[CUDA ✓] {name}")
        return "cuda"
    print("[CPU  ] No CUDA GPU, using CPU")
    return "cpu"


def make_env(seed: int = 0) -> Monitor:
    """
    Build one Monitor-wrapped P3Env.
    env.reset() 每次从全部 200 个 scenario 中随机抽取一个，
    保证训练覆盖所有问题实例。
    """
    random.seed(seed)
    caps, reqs = C.SCENARIOS[0]   # 初始 scenario，reset() 后会随机替换
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(P3Env(ecus, services, scenarios=C.SCENARIOS))


def moving_avg(arr, w: int):
    """Sliding-window average; returns (smoothed, start_offset)."""
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
    Reads Monitor's info["episode"] (written on done=True) for AR,
    and reads the raw info dict for violation counts.

    Monitor wraps reward; episode["r"] == cumulative reward == final AR
    because the only non-zero reward is the last step's AR.
    """

    def __init__(self):
        super().__init__()
        self.episode_ars:        list[float] = []
        self.episode_violations: list[int]   = []   # total violations per episode
        self.timesteps_at_ep:    list[int]   = []
        self._pending_violations: int        = 0    # accumulate within episode

    def _on_step(self) -> bool:
        # Accumulate violations from every step info
        for info in self.locals.get("infos", []):
            # "total_violations" is the *cumulative* count inside the episode.
            # We'll snapshot it at episode end.
            pass

        # Episode end: Monitor injects info["episode"]
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ar = float(info["episode"]["r"])
                # total_violations at episode end = final cumulative count
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

    print(f"\n{'═'*60}")
    print(f"  Problem 3 — PPO (NO constraint enforcement)")
    print(f"  N={C.N}  M={C.M}  steps={C.TOTAL_STEPS:,}  device={device.upper()}")
    print(f"{'═'*60}\n")

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
    log_path = C.OUTDIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved  → {log_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_training_curve(cb, C.OUTDIR)

    env.close()
    print("\nDone. Run evaluate_p3.py to compare against random baseline.\n")


if __name__ == "__main__":
    main()
