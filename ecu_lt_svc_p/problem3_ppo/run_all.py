"""
run_all.py — One-shot P3 full pipeline:

  1. Load a scenario from the same YAML used by problem2_ilp
  2. Solve with ILP (PuLP)                   -> ilp_ar  (optimal upper bound)
  3. Evaluate a random policy                -> random_ars + violations
  4. Train PPO (NO constraint enforcement)   -> training curve
  5. Evaluate the trained PPO agent          -> ppo_ars + violations
  6. Produce two plots:
       - comparison.png    — AR box plot + violation bar chart (3-way)
       - training_curve.png — AR & violation count during training

P3 Design: constraints are RECORDED but NOT enforced.
  - Episodes always run M steps (never terminate early).
  - No penalty for violation; reward = final AR only.

Run:
    python problem3_ppo/run_all.py
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

# ── path setup ────────────────────────────────────────────────────────────────────────────
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
from stable_baselines3.common.vec_env import DummyVecEnv

import config as C
from problem3_ppo.env import P3Env
from problem2_ilp.objects import ECU, SVC
from run_utils import parse_args, resolve_device, moving_avg, solve_ilp, solve_ilp_all_scenarios, load_scenario


def _make_p3_env(seed: int) -> Monitor:
    import random
    random.seed(seed)
    caps, reqs, _ = C.SCENARIOS[C.SCENARIO_IDX]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(P3Env(ecus, services, scenarios=C.TRAIN_SCENARIOS))


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 & 5 — Evaluation (Random / PPO)
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps: int):
    """
    Run n_eps episodes on a fixed problem instance.
    Episodes always complete (M steps, no early termination in P3).
    policy_fn(obs) -> int
    """
    env = P3Env(ecus, services, scenarios=C.TEST_SCENARIOS)
    ars, cap_viols, conflict_viols, viol_rates, placed_list = [], [], [], [], []

    for _ in range(n_eps):
        obs, _ = env.reset()
        done   = False
        info   = {}
        while not done:
            obs, _, done, _, info = env.step(policy_fn(obs))
        ars.append(info["ar"])
        cap_viols.append(info["capacity_violations"])
        conflict_viols.append(info["conflict_violations"])
        viol_rates.append(float(info.get("violation_rate", 0.0)))
        placed_list.append(int(info.get("services_placed", 0)))

    return {
        "ars":            np.array(ars),
        "cap_viols":      np.array(cap_viols),
        "conflict_viols": np.array(conflict_viols),
        "tot_viols":      np.array(cap_viols) + np.array(conflict_viols),
        "viol_rates":     np.array(viol_rates),
        "placed":         np.array(placed_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — PPO training
# ══════════════════════════════════════════════════════════════════════════════

class P3Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_ars             : list[float] = []
        self.episode_viol_rates      : list[float] = []
        self.episode_cap_viol_rates  : list[float] = []
        self.episode_conf_viol_rates : list[float] = []
        self.episode_placed          : list[int]   = []
        self.timesteps_at_ep         : list[int]   = []
        self._next_progress_step = C.PROGRESS_LOG_EVERY_STEPS
        self._t_start = 0.0

    def _on_training_start(self) -> None:
        self._t_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_ars.append(float(info.get("ar", 0.0)))
                self.episode_viol_rates.append(float(info.get("violation_rate", 0.0)))
                self.episode_cap_viol_rates.append(info.get("capacity_violations", 0) / C.M)
                self.episode_conf_viol_rates.append(info.get("conflict_violations", 0) / C.M)
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


def train_ppo(ecus, services, device: str) -> tuple[PPO, P3Callback]:
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = DummyVecEnv(
        [functools.partial(_make_p3_env, C.SEED + i) for i in range(n_envs)],
    )
    print(f"  Using DummyVecEnv: n_envs={n_envs}")
    cb    = P3Callback()
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
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep   = len(cb.episode_ars)
    last50 = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    print(f"  Training done  {elapsed:.1f}s | {n_ep} episodes | AR(last50)={last50:.4f}")
    return model, cb


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curve(cb: P3Callback, ilp_ar: float, outdir: Path, scenario_name: str):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ts = np.array(cb.timesteps_at_ep)

    sm, off = moving_avg(cb.episode_ars, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_ars, color="steelblue", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off:off+len(sm)], sm, color="steelblue", linewidth=2,
             label=f"Method AR (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Training Metrics — {scenario_name}  ({C.TOTAL_STEPS:,} steps)", fontsize=12)
    ax1.grid(alpha=0.3)

    sm_cap,  off_cap  = moving_avg(cb.episode_cap_viol_rates,  C.SMOOTH_W)
    sm_conf, off_conf = moving_avg(cb.episode_conf_viol_rates, C.SMOOTH_W)
    ax2.plot(ts, cb.episode_cap_viol_rates,  color="tomato",    alpha=0.15, linewidth=0.6)
    ax2.plot(ts, cb.episode_conf_viol_rates, color="darkorange", alpha=0.15, linewidth=0.6)
    ax2.plot(ts[off_cap:off_cap+len(sm_cap)],   sm_cap,  color="tomato",    linewidth=2, label="Cap viol rate (smoothed)")
    ax2.plot(ts[off_conf:off_conf+len(sm_conf)], sm_conf, color="darkorange", linewidth=2, label="Conflict viol rate (smoothed)")
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
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
    print(f"  Saved → {path}")


def plot_comparison(ilp_ar, rand_res, ppo_res, ppo_train_viol_mean, ppo_train_viol_std, outdir: Path, scenario_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"P2(ILP) vs Random vs P3(PPO) - {scenario_name}", fontsize=13, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels = ["ILP\n(Optimal)", "Random\nBaseline", "PPO\n(P3, no constraint)"]

    ax = axes[0]
    rand_ars = rand_res["ars"]
    ppo_ars  = ppo_res["ars"]

    bp = ax.boxplot(
        [rand_ars, ppo_ars],
        positions=[2, 3],
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors[1:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(ilp_ar, color=colors[0], linestyle="--", linewidth=2, alpha=0.9,
               label=f"ILP  AR={ilp_ar:.4f}")
    ax.plot(1, ilp_ar, marker="D", color=colors[0], markersize=10, zorder=5)

    for pos, data, color in zip([2, 3], [rand_ars, ppo_ars], colors[1:]):
        mv = np.mean(data)
        ax.text(pos, mv + 0.02, f"μ={mv:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="black")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax.set_title("AR Distribution", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    vr_means = [
        0.0,
        np.mean(rand_res["viol_rates"]),
        ppo_train_viol_mean,
    ]
    vr_stds = [
        0.0,
        np.std(rand_res["viol_rates"]),
        ppo_train_viol_std,
    ]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75,
                   yerr=vr_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 v + max(vr_means) * 0.02 + 0.01,
                 f"{v:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")

    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_title("Violation Rate (eval)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = axes[2]
    pl_means = [C.M, np.mean(rand_res["placed"]), np.mean(ppo_res["placed"])]
    pl_stds = [0.0, np.std(rand_res["placed"]), np.std(ppo_res["placed"])]
    bars = ax3.bar(labels, pl_means, color=colors, alpha=0.75,
                   yerr=pl_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, pl_means):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.05,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")
    ax3.set_ylim(0, C.M + 1)
    ax3.set_ylabel("Services Placed", fontsize=11)
    ax3.set_title("Placement Completeness", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


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
    print(f"  P3 run_all.py  —  RL WITHOUT constraint enforcement")
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}  |  prototype idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    # ── 1. Load scenario ─────────────────────────────────────────────────────
    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N = len(ecus)
    M = len(services)

    # ── 2. ILP (all scenarios) ────────────────────────────────────────────
    print(f"\n[1/4] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    # ── 3. Random baseline ───────────────────────────────────────────────────
    print(f"\n[2/4] Random baseline evaluation ({C.EVAL_EPS} episodes) ...")
    np.random.seed(C.SEED)
    rand_res = run_episodes(
        ecus, services,
        policy_fn=lambda obs: np.random.randint(0, N),
        n_eps=C.EVAL_EPS,
    )
    print(f"  Random AR  mean={np.mean(rand_res['ars']):.4f}  "
          f"std={np.std(rand_res['ars']):.4f}")
    print(f"  Viol rate mean={np.mean(rand_res['viol_rates']):.2%}")

    # ── 4. PPO training ──────────────────────────────────────────────────────
    print(f"\n[3/4] PPO training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_ppo(ecus, services, device)
    model.save(str(C.MODEL_PATH))
    print(f"  Model saved → {C.MODEL_PATH}.zip")

    # ── 5. PPO evaluation ────────────────────────────────────────────────────
    print(f"\n[4/4] PPO evaluation ({C.EVAL_EPS} episodes, deterministic) ...")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ppo_res = run_episodes(ecus, services, ppo_policy, C.EVAL_EPS)
    print(f"  PPO AR  mean={np.mean(ppo_res['ars']):.4f}  "
          f"std={np.std(ppo_res['ars']):.4f}")
    print(f"  Eval viol rate mean={np.mean(ppo_res['viol_rates']):.2%}")

    # ── Summary table ─────────────────────────────────────────────────────────
    M_total = M
    print(f"\n{'='*62}")
    print(f"  {'Method':<24} {'AR (mean±std)':<22} {'ViolRate':<10}")
    print(f"  {'-'*24} {'-'*22} {'-'*10}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} ± 0.0000       {'0':<10}")
    r_v = np.mean(rand_res['viol_rates'])
    p_v = float(np.mean(ppo_res["viol_rates"]))
    p_v_std = float(np.std(ppo_res["viol_rates"]))
    print(f"  {'Random Baseline':<24} "
          f"{np.mean(rand_res['ars']):.4f} ± {np.std(rand_res['ars']):.4f}   "
          f"  {r_v:<10.2%}")
    print(f"  {'PPO (P3, no constr)':<24} "
          f"{np.mean(ppo_res['ars']):.4f} ± {np.std(ppo_res['ars']):.4f}   "
          f"  {p_v:<10.2%}")
    print(f"{'='*62}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    log = {
        "scenario": sc_name,
        "prototype_scenario": prototype_name,
        "scenario_count": len(C.SCENARIOS),
        "train_count": len(C.TRAIN_SCENARIOS),
        "test_count": len(C.TEST_SCENARIOS),
        "N": N, "M": M,
        "ilp":    {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "random": {
            "ar_mean":            round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":             round(float(np.std(rand_res["ars"])),  6),
            "viol_rate_mean":     round(float(r_v), 6),
            "cap_viol_total":     int(np.sum(rand_res["cap_viols"])),
            "conflict_viol_total": int(np.sum(rand_res["conflict_viols"])),
        },
        "ppo": {
            "ar_mean":            round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":             round(float(np.std(ppo_res["ars"])),  6),
            "viol_rate_mean":     round(float(p_v), 6),
            "cap_viol_total":     int(np.sum(ppo_res["cap_viols"])),
            "conflict_viol_total": int(np.sum(ppo_res["conflict_viols"])),
        },
        "training": {
            "total_steps":  C.TOTAL_STEPS,
            "n_episodes":   len(cb.episode_ars),
            "ar_last50":    round(float(np.mean(cb.episode_ars[-50:])), 6),
            "viol_rate_last50": round(float(np.mean(cb.episode_viol_rates[-50:])), 6),
        }
    }
    log_path = C.OUTDIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" / "results.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = log_path.parent
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved → {log_path}")

    # ── Save CSV summary ────────────────────────────────────────────────────────────────────
    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "viol_rate", "cap_viol_total", "conflict_viol_total"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, 0.0, 0, 0])
        writer.writerow([
            "Random Baseline",
            round(float(np.mean(rand_res["ars"])), 6),
            round(float(np.std(rand_res["ars"])), 6),
            round(float(np.mean(rand_res["placed"])), 2),
            round(float(r_v), 4),
            int(np.sum(rand_res["cap_viols"])),
            int(np.sum(rand_res["conflict_viols"])),
        ])
        writer.writerow([
            "PPO (P3, no constr)",
            round(float(np.mean(ppo_res["ars"])), 6),
            round(float(np.std(ppo_res["ars"])), 6),
            round(float(np.mean(ppo_res["placed"])), 2),
            round(float(p_v), 4),
            int(np.sum(ppo_res["cap_viols"])),
            int(np.sum(ppo_res["conflict_viols"])),
        ])
    print(f"  CSV  saved → {csv_path}")
    plot_training_curve(cb, ilp_ar, run_dir, sc_name)
    plot_comparison(ilp_ar, rand_res, ppo_res, p_v, p_v_std, run_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {run_dir}/training_curve.png")
    print(f"  {run_dir}/comparison.png")
    print(f"  {run_dir}/results.json")
    print(f"  {run_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
