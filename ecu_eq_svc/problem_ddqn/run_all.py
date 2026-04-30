"""run_all.py - One-shot DDQN full pipeline:

  1. Load scenario from YAML
  2. Solve with ILP (PuLP)             -> ilp_ar (optimal upper bound)
  3. Evaluate random policy (no mask)  -> random_ars + violations
  4. Train DDQN (no action masking)     -> training curve
  5. Evaluate trained DDQN              -> ddqn_ars + violations
  6. Produce plots:
    - comparison.png     - AR box plot + violation rate bar (3-way)
    - training_curve.png - AR & violation rate during training

DDQN Design: NO action masking.
    - Constraint violations terminate the episode with an unfinished-services penalty.
    - Valid assignments earn their exact utilisation contribution n_i / e_j.
    - Uses off-policy learning with independent target network (similar to DQN).

Run:
    python problem_ddqn/run_all.py
"""

import datetime
import csv
import argparse
import functools
import sys, time, json
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
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import config as C
from problem_ddqn.env import DDQNEnv
from problem2_ilp.objects import ECU, SVC
from run_utils import parse_args, resolve_device, moving_avg, solve_ilp, solve_ilp_all_scenarios, load_scenario


# ── Double DQN ────────────────────────────────────────────────────────────────

class DDQN(DQN):
    """Double DQN: online network selects next action, target network evaluates it.

    SB3's DQN uses q_net_target for both selection and evaluation (overestimation
    bias). Here q_net selects the greedy action and q_net_target provides the value,
    decoupling the two roles per the Double DQN paper (van Hasselt et al., 2016).
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        import torch.nn.functional as F
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = getattr(replay_data, "discounts", None)
            if discounts is None:
                discounts = self.gamma
            with torch.no_grad():
                # Double DQN: online network selects the greedy next action
                best_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # Target network evaluates that action's value
                next_q = self.q_net_target(replay_data.next_observations).gather(1, best_actions)
                target_q = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q
            current_q = self.q_net(replay_data.observations)
            current_q = torch.gather(current_q, dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


def _make_ddqn_env(seed: int) -> Monitor:
    import random
    random.seed(seed)
    caps, reqs = C.SCENARIOS[C.SCENARIO_IDX]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(DDQNEnv(ecus, services, scenarios=C.TRAIN_SCENARIOS))


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 & 5 — Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps):
    """policy_fn(obs) -> int   (no mask)"""
    env = DDQNEnv(ecus, services, scenarios=C.TEST_SCENARIOS)
    ars, placed_list, viol_list = [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            obs, _, done, _, info = env.step(policy_fn(obs))
        ars.append(info.get("ar", 0.0))
        placed_list.append(info.get("services_placed", 0))
        viol_list.append(1 if info.get("violated", False) else 0)
    return {
        "ars":    np.array(ars),
        "placed": np.array(placed_list),
        "viols":  np.array(viol_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Training
# ══════════════════════════════════════════════════════════════════════════════

class DDQNCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards:  list[float] = []
        self.episode_ars:      list[float] = []
        self.episode_placed:   list[int]   = []
        self.episode_violated: list[int]   = []
        self.timesteps_at_ep:  list[int]   = []
        self._next_progress_step = C.PROGRESS_LOG_EVERY_STEPS
        self._t_start = 0.0

    def _on_training_start(self) -> None:
        self._t_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_ars.append(float(info.get("ar", 0.0)))
                self.episode_placed.append(int(info.get("services_placed", 0)))
                self.episode_violated.append(1 if info.get("violated", False) else 0)
                self.timesteps_at_ep.append(self.num_timesteps)

        if self.num_timesteps >= self._next_progress_step:
            elapsed = max(time.time() - self._t_start, 1e-6)
            pct = min(100.0, self.num_timesteps * 100.0 / C.TOTAL_STEPS)
            eps = len(self.episode_rewards)
            sps = self.num_timesteps / elapsed
            print(
                f"  [train] step={self.num_timesteps:,}/{C.TOTAL_STEPS:,} "
                f"({pct:5.1f}%) | eps={eps} | steps/s={sps:,.0f}"
            )
            self._next_progress_step += C.PROGRESS_LOG_EVERY_STEPS
        return True


def train_ddqn(ecus, services, device: str):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = SubprocVecEnv(
        [functools.partial(_make_ddqn_env, C.SEED + i) for i in range(n_envs)],
        start_method=C.SUBPROC_START_METHOD,
    )
    print(f"  Using SubprocVecEnv: n_envs={n_envs}, start_method={C.SUBPROC_START_METHOD}")
    cb = DDQNCallback()
    model = DDQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = C.DDQN_LR,
        buffer_size            = C.DDQN_BUFFER_SIZE,
        learning_starts        = C.DDQN_LEARNING_STARTS,
        batch_size             = C.DDQN_BATCH_SIZE,
        tau                    = C.DDQN_TAU,
        gamma                  = C.DDQN_GAMMA,
        train_freq             = C.DDQN_TRAIN_FREQ,
        gradient_steps         = C.DDQN_GRADIENT_STEPS,
        target_update_interval = C.DDQN_TARGET_UPDATE,
        exploration_fraction   = C.DDQN_EXPLORATION_FRACTION,
        exploration_final_eps  = C.DDQN_EXPLORATION_FINAL_EPS,
        policy_kwargs          = dict(net_arch=C.DDQN_NET_ARCH),
        device                 = device,
        verbose                = 0,
        seed                   = C.SEED,
    )
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep     = len(cb.episode_rewards)
    last50_r = np.mean(cb.episode_rewards[-50:]) if n_ep >= 50 else np.mean(cb.episode_rewards)
    last50_ar = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_v = np.mean(cb.episode_violated[-50:]) if n_ep >= 50 else np.mean(cb.episode_violated)
    print(f"  DDQN Training done  {elapsed:.1f}s | {n_ep} eps "
            f"| reward(last50)={last50_r:.4f} | AR(last50)={last50_ar:.4f}"
            f" | viol_rate(last50)={last50_v:.2%}")
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
             label=f"DDQN AR (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Training Metrics — {scenario_name}  ({C.TOTAL_STEPS:,} steps)", fontsize=12)
    ax1.grid(alpha=0.3)

    viol_rate = np.array(cb.episode_violated, dtype=float)
    sm_v, off_v = moving_avg(viol_rate, C.SMOOTH_W)
    ax2.plot(ts, viol_rate, color="tomato", alpha=0.2, linewidth=0.8)
    ax2.plot(ts[off_v:off_v+len(sm_v)], sm_v, color="tomato", linewidth=2,
             label="violation rate (smoothed)")
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    sm_p, off_p = moving_avg(cb.episode_placed, C.SMOOTH_W)
    ax3.plot(ts, cb.episode_placed, color="royalblue", alpha=0.2, linewidth=0.8)
    ax3.plot(ts[off_p:off_p+len(sm_p)], sm_p, color="royalblue", linewidth=2,
             label="Services placed (smoothed)")
    ax3.axhline(C.M, color="gray", linestyle=":", alpha=0.6, label=f"M={C.M}")
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


def plot_comparison(ilp_ar, rand_res, ddqn_res, ddqn_train_viol_mean, ddqn_train_viol_std, outdir, scenario_name):
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels = ["ILP\n(Optimal)", "Random\n(no mask)", "DDQN\n(no mask)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"ILP vs Random vs DDQN - {scenario_name}", fontsize=13, fontweight="bold")

    # AR box plot
    ax = axes[0]
    bp = ax.boxplot(
        [rand_res["ars"], ddqn_res["ars"]],
        positions=[2, 3], widths=0.5, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors[1:]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.axhline(ilp_ar, color=colors[0], linestyle="--", linewidth=2, alpha=0.9,
               label=f"ILP  AR={ilp_ar:.4f}")
    ax.plot(1, ilp_ar, marker="D", color=colors[0], markersize=10, zorder=5)

    for pos, data, color in zip([2, 3], [rand_res["ars"], ddqn_res["ars"]], colors[1:]):
        mv = np.mean(data)
        ax.text(pos, mv + 0.02, f"mu={mv:.3f}", ha="center", fontsize=9,
                fontweight="bold", color="black")

    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax.set_title("Episode-end AR Distribution", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    vr_means = [0.0, np.mean(rand_res["viols"]), ddqn_train_viol_mean]
    vr_stds  = [0.0, np.std(rand_res["viols"]),  ddqn_train_viol_std]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75,
                   yerr=vr_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color="black")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_title("Violation Rate (eval)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = axes[2]
    pl_means = [C.M, np.mean(rand_res["placed"]), np.mean(ddqn_res["placed"])]
    pl_stds = [0.0, np.std(rand_res["placed"]), np.std(ddqn_res["placed"])]
    bars = ax3.bar(labels, pl_means, color=colors, alpha=0.75,
                   yerr=pl_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, pl_means):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.05,
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
    args = parse_args()
    if args.total_timesteps is not None:
        C.TOTAL_STEPS = int(args.total_timesteps)
        print(f"[override] TOTAL_STEPS={C.TOTAL_STEPS:,}")

    base_dir = C.OUTDIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DDQN run_all.py  — RL WITHOUT action masking")
    print(f"  Violation \u2192 reward=-1, episode terminates immediately.")
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}  |  prototype idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    # 1. Load scenario
    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    # 2. ILP (all scenarios)
    print(f"\n[1/4] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    # 3. Random baseline (no masking)
    print(f"\n[2/4] Random baseline ({C.EVAL_EPS} episodes, NO masking) ...")
    np.random.seed(C.SEED)
    rand_res = run_episodes(
        ecus, services,
        policy_fn=lambda obs: int(np.random.randint(0, N)),
        n_eps=C.EVAL_EPS,
    )
    print(f"  Random AR  mean={np.mean(rand_res['ars']):.4f}  "
          f"std={np.std(rand_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(rand_res['placed']):.1f}/{M}")
    print(f"  Viol rate  {np.mean(rand_res['viols']):.2%}")

    # 4. DDQN training
    print(f"\n[3/4] DDQN training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_ddqn(ecus, services, device)
    model_path = base_dir / "ddqn_model"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    # 5. DDQN evaluation
    print(f"\n[4/4] DDQN evaluation ({C.EVAL_EPS} episodes, deterministic) ...")
    def ddqn_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ddqn_res = run_episodes(ecus, services, ddqn_policy, C.EVAL_EPS)
    print(f"  DDQN AR  mean={np.mean(ddqn_res['ars']):.4f}  "
          f"std={np.std(ddqn_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(ddqn_res['placed']):.1f}/{M}")
    print(f"  Eval viol rate  {np.mean(ddqn_res['viols']):.2%}")

    ddqn_train_v = float(np.mean(ddqn_res["viols"]))
    ddqn_train_v_std = float(np.std(ddqn_res["viols"]))

    # Summary
    print(f"\n{'='*68}")
    print(f"  {'Method':<24} {'AR (mean+/-std)':<22} {'Placed':<10} {'Viol%'}")
    print(f"  {'-'*24} {'-'*22} {'-'*10} {'-'*6}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} +/- 0.0000     {M}/{M:<6} 0%")
    print(f"  {'Random (no mask)':<24} "
          f"{np.mean(rand_res['ars']):.4f} +/- {np.std(rand_res['ars']):.4f}   "
          f"  {np.mean(rand_res['placed']):.1f}/{M:<2}   {np.mean(rand_res['viols']):.0%}")
    print(f"  {'DDQN (no mask)':<24} "
          f"{np.mean(ddqn_res['ars']):.4f} +/- {np.std(ddqn_res['ars']):.4f}   "
            f"{np.mean(ddqn_res['placed']):.1f}/{M:<4} {ddqn_train_v:.1%}")
    print(f"{'='*68}\n")

    # Save JSON
    log = {
        "scenario": sc_name,
        "prototype_scenario": prototype_name,
        "scenario_count": len(C.SCENARIOS),
        "train_count": len(C.TRAIN_SCENARIOS),
        "test_count": len(C.TEST_SCENARIOS),
        "N": N,
        "M": M,
        "ilp": {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "random": {
            "ar_mean":     round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":      round(float(np.std(rand_res["ars"])), 6),
            "placed_mean": round(float(np.mean(rand_res["placed"])), 2),
            "viol_rate":   round(float(np.mean(rand_res["viols"])), 4),
        },
        "ddqn": {
            "ar_mean":     round(float(np.mean(ddqn_res["ars"])), 6),
            "ar_std":      round(float(np.std(ddqn_res["ars"])), 6),
            "placed_mean": round(float(np.mean(ddqn_res["placed"])), 2),
            "viol_rate":   round(float(ddqn_train_v), 4),
        },
        "training": {
            "total_steps":      C.TOTAL_STEPS,
            "n_episodes":       len(cb.episode_rewards),
            "ar_last50":        round(float(np.mean(cb.episode_ars[-50:])), 6),
            "reward_last50":    round(float(np.mean(cb.episode_rewards[-50:])), 6),
            "viol_rate_last50": round(float(np.mean(cb.episode_violated[-50:])), 4),
        },
    }
    log_path = base_dir / "results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {log_path}")

    # Save CSV summary
    csv_path = base_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "viol_rate"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, 0.0])
        writer.writerow([
            "Random (no mask)",
            round(float(np.mean(rand_res["ars"])), 6),
            round(float(np.std(rand_res["ars"])), 6),
            round(float(np.mean(rand_res["placed"])), 2),
            round(float(np.mean(rand_res["viols"])), 4),
        ])
        writer.writerow([
            "DDQN (no mask)",
            round(float(np.mean(ddqn_res["ars"])), 6),
            round(float(np.std(ddqn_res["ars"])), 6),
            round(float(np.mean(ddqn_res["placed"])), 2),
            round(float(ddqn_train_v), 4),
        ])
    print(f"  CSV  saved -> {csv_path}")

    plot_training_curve(cb, ilp_ar, base_dir, sc_name)
    plot_comparison(ilp_ar, rand_res, ddqn_res, ddqn_train_v, ddqn_train_v_std, base_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {base_dir}/training_curve.png")
    print(f"  {base_dir}/comparison.png")
    print(f"  {base_dir}/results.json")
    print(f"  {base_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
