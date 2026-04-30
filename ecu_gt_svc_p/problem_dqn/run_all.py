"""
run_all.py — One-shot DQN full pipeline:

  1. Load scenario from YAML
  2. Solve with ILP (PuLP)             -> ilp_ar (optimal upper bound)
  3. Train DQN (no action masking)     -> training curve
  4. Evaluate trained DQN              -> dqn_ars + violations
  5. Produce plots:
    - comparison.png     — AR box plot + violation rate bar (2-way: ILP vs DQN)
    - training_curve.png — AR & violation rate during training

DQN Design: NO action masking.
    - Constraint violations terminate the episode with an unfinished-services penalty.
    - Valid assignments earn their exact utilisation contribution n_i / e_j.

Run:
    python dqn/run_all.py
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
from stable_baselines3.common.vec_env import DummyVecEnv

import config as C
from problem_dqn.env import DQNEnv
from problem2_ilp.objects import ECU, SVC
from run_utils import parse_args, resolve_device, moving_avg, solve_ilp, solve_ilp_all_scenarios, load_scenario


def _make_dqn_env(seed: int) -> Monitor:
    import random
    random.seed(seed)
    caps, reqs, _ = C.SCENARIOS[C.SCENARIO_IDX]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(DQNEnv(ecus, services, scenarios=C.TRAIN_SCENARIOS))


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 & 5 — Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps):
    """policy_fn(obs) -> int   (no mask)"""
    env = DQNEnv(ecus, services, scenarios=C.TEST_SCENARIOS)
    ars, placed_list, viol_list, conflict_viol_list = [], [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            obs, _, done, _, info = env.step(policy_fn(obs))
        ars.append(info.get("ar", 0.0))
        placed_list.append(info.get("services_placed", 0))
        viol_list.append(1 if info.get("violated", False) else 0)
        conflict_viol_list.append(int(info.get("conflict_violations", 0)))
    return {
        "ars":           np.array(ars),
        "placed":        np.array(placed_list),
        "viols":         np.array(viol_list),
        "conflict_viols": np.array(conflict_viol_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Training
# ══════════════════════════════════════════════════════════════════════════════

class DQNCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards:      list[float] = []
        self.episode_ars:          list[float] = []
        self.episode_placed:       list[int]   = []
        self.episode_violated:     list[int]   = []
        self.episode_cap_violated:  list[int]  = []
        self.episode_conf_violated: list[int]  = []
        self.timesteps_at_ep:      list[int]   = []
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
                violated     = info.get("violated", False)
                conf_viols   = int(info.get("conflict_violations", 0))
                self.episode_violated.append(1 if violated else 0)
                self.episode_cap_violated.append(1 if (violated and conf_viols == 0) else 0)
                self.episode_conf_violated.append(1 if conf_viols > 0 else 0)
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


def train_dqn(ecus, services, device: str):
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = DummyVecEnv(
        [functools.partial(_make_dqn_env, C.SEED + i) for i in range(n_envs)],
    )
    print(f"  Using DummyVecEnv: n_envs={n_envs}")
    cb = DQNCallback()
    model = DQN(
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
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep     = len(cb.episode_rewards)
    last50_r = np.mean(cb.episode_rewards[-50:]) if n_ep >= 50 else np.mean(cb.episode_rewards)
    last50_ar = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_v = np.mean(cb.episode_violated[-50:]) if n_ep >= 50 else np.mean(cb.episode_violated)
    print(f"  Training done  {elapsed:.1f}s | {n_ep} eps "
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
             label=f"DQN AR (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Training Metrics — {scenario_name}  ({C.TOTAL_STEPS:,} steps)", fontsize=12)
    ax1.grid(alpha=0.3)

    cap_rate  = np.array(cb.episode_cap_violated,  dtype=float)
    conf_rate = np.array(cb.episode_conf_violated, dtype=float)
    sm_cap,  off_cap  = moving_avg(cap_rate,  C.SMOOTH_W)
    sm_conf, off_conf = moving_avg(conf_rate, C.SMOOTH_W)
    ax2.plot(ts, cap_rate,  color="tomato",    alpha=0.15, linewidth=0.6)
    ax2.plot(ts, conf_rate, color="darkorange", alpha=0.15, linewidth=0.6)
    ax2.plot(ts[off_cap:off_cap+len(sm_cap)],   sm_cap,  color="tomato",    linewidth=2, label="Cap viol rate (smoothed)")
    ax2.plot(ts[off_conf:off_conf+len(sm_conf)], sm_conf, color="darkorange", linewidth=2, label="Conflict viol rate (smoothed)")
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


def plot_comparison(ilp_ar, dqn_res, dqn_train_viol_mean, dqn_train_viol_std, outdir, scenario_name):
    colors = ["#e74c3c", "#2ecc71"]
    labels = ["ILP\n(Optimal)", "DQN\n(no mask)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"ILP vs DQN - {scenario_name}", fontsize=13, fontweight="bold")

    # AR box plot
    ax = axes[0]
    bp = ax.boxplot(
        [dqn_res["ars"]],
        positions=[2], widths=0.5, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors[1:]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.axhline(ilp_ar, color=colors[0], linestyle="--", linewidth=2, alpha=0.9,
               label=f"ILP  AR={ilp_ar:.4f}")
    ax.plot(1, ilp_ar, marker="D", color=colors[0], markersize=10, zorder=5)

    mv = np.mean(dqn_res["ars"])
    ax.text(2, mv + 0.02, f"mu={mv:.3f}", ha="center", fontsize=9,
            fontweight="bold", color="black")

    ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax.set_title("Episode-end AR Distribution", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    vr_means = [0.0, dqn_train_viol_mean]
    vr_stds  = [0.0, dqn_train_viol_std]
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
    pl_means = [C.M, np.mean(dqn_res["placed"])]
    pl_stds  = [0.0, np.std(dqn_res["placed"])]
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
    print(f"  DQN run_all.py  \u2014  RL WITHOUT action masking")
    print(f"  Violation \u2192 reward=-1, episode terminates immediately.")
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}  |  prototype idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    # 1. Load scenario
    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N, M = len(ecus), len(services)

    from run_utils import check_scenario_feasibility
    print("\n[Feasibility Check]")
    train_feas = check_scenario_feasibility(C.TRAIN_SCENARIOS)
    test_feas  = check_scenario_feasibility(C.TEST_SCENARIOS)
    print(f"  Train: {train_feas['feasible']}/{train_feas['total']} feasible, {train_feas['infeasible']} infeasible (idx: {train_feas['infeasible_indices']})")
    print(f"  Test:  {test_feas['feasible']}/{test_feas['total']} feasible, {test_feas['infeasible']} infeasible (idx: {test_feas['infeasible_indices']})")

    # 2. ILP (all scenarios)
    print(f"\n[1/3] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    # 3. DQN training
    print(f"\n[2/3] DQN training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_dqn(ecus, services, device)
    model_path = base_dir / "dqn_model"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    # 4. DQN evaluation
    print(f"\n[3/3] DQN evaluation ({C.EVAL_EPS} episodes, deterministic) ...")
    def dqn_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    dqn_res = run_episodes(ecus, services, dqn_policy, C.EVAL_EPS)
    print(f"  DQN AR  mean={np.mean(dqn_res['ars']):.4f}  "
          f"std={np.std(dqn_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(dqn_res['placed']):.1f}/{M}")
    print(f"  Eval viol rate  {np.mean(dqn_res['viols']):.2%}")

    dqn_train_v = float(np.mean(dqn_res["viols"]))
    dqn_train_v_std = float(np.std(dqn_res["viols"]))

    # Summary
    print(f"\n{'='*68}")
    print(f"  {'Method':<24} {'AR (mean+/-std)':<22} {'Placed':<10} {'Viol%'}")
    print(f"  {'-'*24} {'-'*22} {'-'*10} {'-'*6}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} +/- 0.0000     {M}/{M:<6} 0%")
    print(f"  {'DQN (no mask)':<24} "
          f"{np.mean(dqn_res['ars']):.4f} +/- {np.std(dqn_res['ars']):.4f}   "
            f"  {np.mean(dqn_res['placed']):.1f}/{M:<2}   {dqn_train_v:.0%}")
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
        "feasibility": {
            "train_feasible": train_feas["feasible"],
            "train_infeasible": train_feas["infeasible"],
            "train_infeasible_indices": train_feas["infeasible_indices"],
            "test_feasible": test_feas["feasible"],
            "test_infeasible": test_feas["infeasible"],
            "test_infeasible_indices": test_feas["infeasible_indices"],
        },
        "ilp": {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "dqn": {
            "ar_mean":            round(float(np.mean(dqn_res["ars"])), 6),
            "ar_std":             round(float(np.std(dqn_res["ars"])), 6),
            "placed_mean":        round(float(np.mean(dqn_res["placed"])), 2),
            "viol_rate":          round(float(dqn_train_v), 4),
            "cap_viol_total":     int(np.sum(dqn_res["viols"])),
            "conflict_viol_total": int(np.sum(dqn_res["conflict_viols"])),
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
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "viol_rate", "cap_viol_total", "conflict_viol_total"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, 0.0, 0, 0])
        writer.writerow([
            "DQN (no mask)",
            round(float(np.mean(dqn_res["ars"])), 6),
            round(float(np.std(dqn_res["ars"])), 6),
            round(float(np.mean(dqn_res["placed"])), 2),
            round(float(dqn_train_v), 4),
            int(np.sum(dqn_res["viols"])),
            int(np.sum(dqn_res["conflict_viols"])),
        ])
    print(f"  CSV  saved -> {csv_path}")

    plot_training_curve(cb, ilp_ar, base_dir, sc_name)
    plot_comparison(ilp_ar, dqn_res, dqn_train_v, dqn_train_v_std, base_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {base_dir}/training_curve.png")
    print(f"  {base_dir}/comparison.png")
    print(f"  {base_dir}/results.json")
    print(f"  {base_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
