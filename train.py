"""
Train a service placement environment with PPO / DQN
• 5 orders of magnitude: N ∈ { 10¹, 10², 10³, 10⁴, 10⁵ } ECUs
• Auto-detect CUDA for GPU acceleration (device="auto")
• Output 3 comparison plots (box plot / training curve / final AR comparison)
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # Save figures without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

# ── Add env directory to path ─────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
from env import my_env, my_ecu, my_service

# ══════════════════════════════════════════════════════════════
#  ★ Configuration — edit only here
# ══════════════════════════════════════════════════════════════

SCALES       = [10, 100, 1_000, 10_000, 100_000]  # N = number of ECUs (5 orders of magnitude)
M_FIXED      = 100          # services per episode (fixed, ensures comparability across scales)
VMS_RANGE    = (10, 101)    # ECU VM capacity range [10, 100] (uniform integer sampling)
REQ_RANGE    = (1,  10)     # service VM demand range [1, 9]; max=9 < VMS min=10 -> never over capacity
ALGO         = "PPO"        # "PPO" or "DQN"
TOTAL_STEPS  = 100_000      # total training steps per scale (100k / 100 steps ≈ 1000 episodes)
PPO_MAX_N    = 110_000       # N above this only runs random baseline (action space too large)
EVAL_EPS     = 300          # number of episodes for random baseline evaluation
DEVICE       = "auto"       # "auto"=auto select CUDA/CPU | "cuda" | "cpu"
SEED         = 42
SMOOTH_W     = 30           # sliding window size for training curve smoothing

OUTDIR = Path(__file__).parent / "results"
os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  Device detection
# ══════════════════════════════════════════════════════════════

def resolve_device(cfg: str) -> str:
    """
    "auto" -> prefer CUDA (GPU), fall back to CPU.
    Forward/backward passes of the PPO/DQN policy network run on this device.
    Large N (N=10000): GPU significantly accelerates the final Linear(128 -> N) layer.
    Small N (N=10~100): CPU is actually faster (GPU data-transfer overhead > compute gain).
    """
    if cfg != "auto":
        return cfg
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[CUDA ✓] {name}  VRAM {mem:.1f} GB")
        return "cuda"
    print("[CPU  ] No CUDA GPU detected, using CPU")
    return "cpu"


# ══════════════════════════════════════════════════════════════
#  Environment factory
# ══════════════════════════════════════════════════════════════

def make_env(N: int, seed: int = 0) -> Monitor:
    """
    Create a Monitor-wrapped environment.
    Monitor automatically logs total reward / length / time per episode
    and injects info["episode"] for the callback to read.
    Since the env only gives reward = AR on the final step (0 elsewhere),
    episode_reward = final_AR and can be used directly as the learning curve metric.
    """
    rng      = np.random.default_rng(seed)
    ecus     = [my_ecu(int(rng.integers(*VMS_RANGE))) for _ in range(N)]
    services = [my_service(i, int(rng.integers(*REQ_RANGE))) for i in range(M_FIXED)]
    return Monitor(my_env(ecus, services))


# ══════════════════════════════════════════════════════════════
#  Callback: collect final AR for each episode
# ══════════════════════════════════════════════════════════════

class ARCallback(BaseCallback):
    """
    PPO/DQN calls _on_step() after every step.
    When Monitor detects done=True it writes info["episode"]:
        {"r": total_reward, "l": episode_length, "t": wall_time}
    We simply read AR from there.
    """

    def __init__(self):
        super().__init__()
        self.episode_ars:     list[float] = []
        self.timesteps_at_ep: list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ar = float(info["episode"]["r"])       # = final AR
                self.episode_ars.append(ar)
                self.timesteps_at_ep.append(self.num_timesteps)
        return True  # returning False would stop training early


# ══════════════════════════════════════════════════════════════
#  Random baseline evaluation
# ══════════════════════════════════════════════════════════════

def eval_random(N: int, n_eps: int = EVAL_EPS) -> np.ndarray:
    """Run n_eps complete episodes for scale N and record the final AR of each."""
    env = make_env(N, seed=SEED)
    ars = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        while not done:
            obs, r, done, _, _ = env.step(env.action_space.sample())
            ep_r += r
        ars.append(ep_r)
    env.close()
    return np.array(ars, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  Build model
# ══════════════════════════════════════════════════════════════

def build_model(algo: str, env: Monitor, device: str, N: int):
    """
    PPO: on-policy, policy gradient. Suitable for continuous control and small discrete action spaces.
    DQN: off-policy, Q-learning + replay buffer. More sample-efficient for discrete actions.
    Both have a final Linear(128, N) layer; CUDA acceleration is most significant for large N.
    """
    if algo == "PPO":
        return PPO(
            "MlpPolicy", env,
            learning_rate = 3e-4,
            n_steps       = 512,        # collect 512 steps per rollout before updating
            batch_size    = 64,
            n_epochs      = 10,         # reuse each batch 10 times
            gamma         = 0.99,       # discount factor (reward only at episode end, needs ~1)
            policy_kwargs = dict(net_arch=[128, 128]),
            device        = device,
            verbose       = 0,
            seed          = SEED,
        )
    else:  # DQN
        return DQN(
            "MlpPolicy", env,
            learning_rate          = 1e-4,
            buffer_size            = 50_000,   # replay buffer size
            learning_starts        = 2_000,    # collect 2000 steps before learning starts
            batch_size             = 64,
            gamma                  = 0.99,
            train_freq             = 4,        # update Q-network every 4 steps
            target_update_interval = 500,      # sync target network every 500 steps
            exploration_fraction   = 0.3,      # linearly decay epsilon over first 30% of steps
            policy_kwargs          = dict(net_arch=[128, 128]),
            device                 = device,
            verbose                = 0,
            seed                   = SEED,
        )


# ══════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════

def train_agent(N: int, device: str) -> "ARCallback | None":
    if N > PPO_MAX_N:
        print(f"  ⚠  N={N:,} exceeds PPO_MAX_N={PPO_MAX_N:,}, skipping training (random baseline only)")
        return None

    env   = make_env(N, seed=SEED)
    cb    = ARCallback()
    model = build_model(ALGO, env, device, N)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb, progress_bar=False)
    elapsed = time.time() - t0

    n_ep    = len(cb.episode_ars)
    last_ar = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    rand_ar = np.mean(eval_random(N, n_eps=100))
    gain    = last_ar - rand_ar

    print(f"  elapsed {elapsed:.1f}s  |  episodes={n_ep}  "
          f"|  last50ep AR={last_ar:.4f}  |  vs random {rand_ar:.4f}  "
          f"|  gain {gain:+.4f}")
    env.close()
    return cb


# ══════════════════════════════════════════════════════════════
#  Utility: sliding average
# ══════════════════════════════════════════════════════════════

def moving_avg(arr: list | np.ndarray, w: int = SMOOTH_W):
    """Return a smoothed array of length len(arr)-w+1 and the corresponding start offset."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    sm = np.convolve(arr, np.ones(w) / w, mode="valid")
    return sm, w - 1           # sm[i] is the mean of arr[i .. i+w-1]


# ══════════════════════════════════════════════════════════════
#  Plot (3 subplots)
# ══════════════════════════════════════════════════════════════

def plot_results(results: dict, device: str):
    COLORS  = plt.cm.viridis(np.linspace(0.1, 0.85, len(SCALES)))
    XLABELS = [f"$10^{{{i+1}}}$" for i in range(len(SCALES))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── ① Random baseline AR box plot ──────────────────────────────
    ax = axes[0]
    bp = ax.boxplot(
        [results[N]["rand"] for N in SCALES],
        labels=XLABELS,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.55),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.3),
    )
    # Annotate mean above median line
    for i, N in enumerate(SCALES):
        mean_val = float(np.mean(results[N]["rand"]))
        ax.text(i + 1, mean_val + 0.03, f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="navy")
    ax.set_title("① Random Policy AR Distribution (per scale)", fontsize=12)
    ax.set_xlabel("Number of ECUs (N, order of magnitude)")
    ax.set_ylabel("AR (resource utilization)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # ── ② PPO/DQN training curve ─────────────────────────────────
    ax = axes[1]
    has_curve = False
    for i, N in enumerate(SCALES):
        cb = results[N]["cb"]
        if cb is None or len(cb.episode_ars) < SMOOTH_W + 5:
            ax.axhline(np.mean(results[N]["rand"]),
                       color=COLORS[i], linestyle="--", linewidth=1,
                       alpha=0.5, label=f"N={N:,} (random baseline only)")
            continue
        has_curve = True
        sm, offset = moving_avg(cb.episode_ars, SMOOTH_W)
        ts = np.array(cb.timesteps_at_ep)[offset: offset + len(sm)]
        ax.plot(ts, sm, color=COLORS[i], label=f"N={N:,}", linewidth=2)

    ax.set_title(f"② {ALGO} Training Curve (sliding mean w={SMOOTH_W})", fontsize=12)
    ax.set_xlabel("Total training steps")
    ax.set_ylabel("AR (episode reward)")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    # ── ③ Final AR comparison bar chart ─────────────────────────
    ax     = axes[2]
    x      = np.arange(len(SCALES))
    bw     = 0.32
    rand_m = [float(np.mean(results[N]["rand"])) for N in SCALES]
    ppo_m  = []
    for N in SCALES:
        cb = results[N]["cb"]
        if cb and len(cb.episode_ars) >= 50:
            ppo_m.append(float(np.mean(cb.episode_ars[-50:])))
        else:
            ppo_m.append(float("nan"))

    bars_r = ax.bar(x - bw / 2, rand_m, bw, label="Random baseline",
                    color="steelblue", alpha=0.75)
    bars_p = ax.bar(x + bw / 2, ppo_m,  bw, label=ALGO,
                    color="coral",     alpha=0.85)

    # Annotate values on top of each bar
    for bar, v in zip(bars_r, rand_m):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, color="steelblue")
    for bar, v in zip(bars_p, ppo_m):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, color="tomato")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "N/A",
                    ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS)
    ax.set_xlabel("Number of ECUs (N, order of magnitude)")
    ax.set_ylabel("Mean AR (last 50 episodes)")
    ax.set_title(f"③ Random vs {ALGO}  Final AR Comparison", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 总标题
    fig.suptitle(
        f"Service Placement RL  ·  {ALGO}  ·  Device={device.upper()}"
        f"  ·  M={M_FIXED} services/episode  ·  {TOTAL_STEPS:,} steps/scale",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(OUTDIR, "training_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved -> {path}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    device = resolve_device(DEVICE)
    print(f"\nAlgo={ALGO}  Device={device.upper()}  M={M_FIXED}  TOTAL_STEPS={TOTAL_STEPS:,}")
    print(f"ECU capacity range={VMS_RANGE}  Service demand range={REQ_RANGE}")
    print(f"{'═' * 60}\n")

    results = {}
    for i, N in enumerate(SCALES):
        print(f"[{i+1}/{len(SCALES)}]  N={N:,} ECU  M={M_FIXED} services")

        # ① Random baseline
        rand_ars = eval_random(N)
        print(f"  Random baseline  mean={np.mean(rand_ars):.4f}  "
              f"std={np.std(rand_ars):.4f}  max={np.max(rand_ars):.4f}")

        # ② Train PPO / DQN
        cb = train_agent(N, device)
        results[N] = {"rand": rand_ars, "cb": cb}
        print()

    # ③ Plot
    plot_results(results, device)
    print("All done!")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.1f} seconds")
    print(f"\nTotal elapsed time: {elapsed/60:.1f} minutes")
    print(f"\nTotal elapsed time: {elapsed/3600:.2f} hours")
