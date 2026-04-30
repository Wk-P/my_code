"""
run_all.py — One-shot P6 full pipeline:

  1. Load a scenario from the same YAML used by problem2_ilp
  2. Solve with ILP (PuLP)               -> ilp_ar  (optimal upper bound)
  3. Train PPO + repair heuristic        -> training curve
  4. Evaluate the trained PPO agent      -> ppo_ars + repair_rates
  5. Produce two plots:
       - comparison.png      — AR box plot + repair rate bar (2-way: ILP vs PPO+Repair)
       - training_curve.png  — AR & repair rate during training

P6 Design: standard PPO + best-fit repair heuristic.
  - Agent selects any ECU; on violation the env repairs to best-fit valid ECU.
  - Repair incurs -0.1 penalty to encourage the agent to learn valid placements.
  - If no repair is possible, episode terminates with demand penalty.
  - Terminal bonus: +AR * (1 - repair_rate).

Run:
    python problem6_ppo_opt/run_all.py
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import config as C
from problem6_ppo_opt.env import P6Env
from problem2_ilp.objects import ECU, SVC
from run_utils import parse_args, resolve_device, moving_avg, solve_ilp, solve_ilp_all_scenarios, load_scenario, check_scenario_feasibility


def _make_p6_env(seed: int) -> Monitor:
    import random
    random.seed(seed)
    caps, reqs, _ = C.SCENARIOS[C.SCENARIO_IDX]
    ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return Monitor(P6Env(ecus, services, scenarios=C.TRAIN_SCENARIOS))


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps: int):
    """policy_fn(obs) -> int"""
    env = P6Env(ecus, services, scenarios=C.TEST_SCENARIOS)
    ars, repair_rates, placed_list = [], [], []
    cap_viol_list, conflict_viol_list = [], []

    for _ in range(n_eps):
        obs, _ = env.reset()
        done   = False
        info   = {}
        while not done:
            obs, _, done, _, info = env.step(policy_fn(obs))
        ars.append(info.get("ar", 0.0))
        repair_rates.append(float(info.get("repair_rate", 0.0)))
        placed_list.append(int(info.get("services_placed", 0)))
        cap_viol_list.append(int(info.get("cap_violations", 0)))
        conflict_viol_list.append(int(info.get("conflict_violations", 0)))

    return {
        "ars":              np.array(ars),
        "repair_rates":     np.array(repair_rates),
        "placed":           np.array(placed_list),
        "cap_violations":   np.array(cap_viol_list),
        "conflict_violations": np.array(conflict_viol_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PPO training
# ══════════════════════════════════════════════════════════════════════════════

class P6Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_ars              : list[float] = []
        self.episode_repair_rates     : list[float] = []
        self.episode_placed           : list[int]   = []
        self.episode_cap_violations      : list[int]   = []
        self.episode_conflict_violations : list[int]   = []
        self.timesteps_at_ep          : list[int]   = []
        self._next_progress_step = C.PROGRESS_LOG_EVERY_STEPS
        self._t_start = 0.0

    def _on_training_start(self) -> None:
        self._t_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_ars.append(float(info.get("ar", 0.0)))
                self.episode_repair_rates.append(float(info.get("repair_rate", 0.0)))
                self.episode_placed.append(int(info.get("services_placed", 0)))
                self.episode_cap_violations.append(int(info.get("cap_violations", 0)))
                self.episode_conflict_violations.append(int(info.get("conflict_violations", 0)))
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


def train_ppo(ecus, services, device: str) -> tuple[PPO, P6Callback]:
    import torch as _torch
    _torch.set_num_threads(C.TORCH_NUM_THREADS)
    sys.stdout.flush()
    n_envs = max(1, int(C.N_ENVS))
    env = SubprocVecEnv(
        [functools.partial(_make_p6_env, C.SEED + i) for i in range(n_envs)],
        start_method=C.SUBPROC_START_METHOD,
    )
    print(f"  Using SubprocVecEnv: n_envs={n_envs}, start_method={C.SUBPROC_START_METHOD}")
    cb    = P6Callback()
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

def plot_training_curve(cb: P6Callback, ilp_ar: float, outdir: Path, scenario_name: str):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ts = np.array(cb.timesteps_at_ep)

    sm, off = moving_avg(cb.episode_ars, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_ars, color="steelblue", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off:off+len(sm)], sm, color="steelblue", linewidth=2,
             label=f"PPO+Repair AR (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Training Metrics — {scenario_name}  ({C.TOTAL_STEPS:,} steps)", fontsize=12)
    ax1.grid(alpha=0.3)

    sm_r, off_r = moving_avg(cb.episode_repair_rates, C.SMOOTH_W)
    ax2.plot(ts, cb.episode_repair_rates, color="darkorange", alpha=0.15, linewidth=0.6)
    ax2.plot(ts[off_r:off_r+len(sm_r)], sm_r, color="darkorange", linewidth=2,
             label="Repair rate (smoothed)")
    ax2.set_ylabel("Repair Rate", fontsize=11)
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


def plot_comparison(ilp_ar, ppo_res, ppo_train_repair_mean, ppo_train_repair_std,
                    outdir: Path, scenario_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"ILP vs PPO+Repair (P6) — {scenario_name}", fontsize=13, fontweight="bold")

    colors = ["#e74c3c", "#2ecc71"]
    labels = ["ILP\n(Optimal)", "PPO+Repair\n(P6)"]

    ax = axes[0]
    ppo_ars = ppo_res["ars"]
    bp = ax.boxplot(
        [ppo_ars],
        positions=[2],
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
    mv = np.mean(ppo_ars)
    ax.text(2, mv + 0.02, f"μ={mv:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax.set_title("AR Distribution", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    rr_means = [0.0, ppo_train_repair_mean]
    rr_stds  = [0.0, ppo_train_repair_std]
    bars = ax2.bar(labels, rr_means, color=colors, alpha=0.75,
                   yerr=rr_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, rr_means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 v + max(rr_means) * 0.02 + 0.01,
                 f"{v:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Repair Rate", fontsize=11)
    ax2.set_title("Repair Rate (eval)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = axes[2]
    pl_means = [C.M, np.mean(ppo_res["placed"])]
    pl_stds  = [0.0, np.std(ppo_res["placed"])]
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
    print(f"  P6 run_all.py  —  PPO + Best-fit Repair Heuristic")
    print(f"  Config : {C.YAML_CONFIG.name}  |  train={len(C.TRAIN_SCENARIOS)}/test={len(C.TEST_SCENARIOS)}  |  prototype idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")
    device = resolve_device(C.DEVICE)

    # ── 1. Load scenario ─────────────────────────────────────────────────────
    ecus, services, sc_name, prototype_name = load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)
    N = len(ecus)
    M = len(services)

    print("\n[Feasibility Check]")
    train_feas = check_scenario_feasibility(C.TRAIN_SCENARIOS)
    test_feas  = check_scenario_feasibility(C.TEST_SCENARIOS)
    print(f"  Train: {train_feas['feasible']}/{train_feas['total']} feasible, {train_feas['infeasible']} infeasible (idx: {train_feas['infeasible_indices']})")
    print(f"  Test:  {test_feas['feasible']}/{test_feas['total']} feasible, {test_feas['infeasible']} infeasible (idx: {test_feas['infeasible_indices']})")

    # ── 2. ILP (all test scenarios) ───────────────────────────────────────────
    print(f"\n[1/3] Solving ILP for {len(C.TEST_SCENARIOS)} test scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios(C.YAML_CONFIG, C.TEST_SCENARIOS, C.OUTDIR)
    print(f"  ILP mean AR across {len(C.TEST_SCENARIOS)} test scenarios: {ilp_ar:.4f}")

    # ── 3. PPO training ──────────────────────────────────────────────────────
    print(f"\n[2/3] PPO training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_ppo(ecus, services, device)
    model.save(str(C.MODEL_PATH))
    print(f"  Model saved → {C.MODEL_PATH}.zip")

    # ── 4. PPO evaluation ────────────────────────────────────────────────────
    print(f"\n[3/3] PPO evaluation ({C.EVAL_EPS} episodes, deterministic) ...")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ppo_res = run_episodes(ecus, services, ppo_policy, C.EVAL_EPS)
    p_rr      = float(np.mean(ppo_res["repair_rates"]))
    p_rr_std  = float(np.std(ppo_res["repair_rates"]))
    p_cap_viol = float(np.sum(ppo_res["cap_violations"]))
    p_con_viol = float(np.sum(ppo_res["conflict_violations"]))
    print(f"  PPO AR           mean={np.mean(ppo_res['ars']):.4f}  std={np.std(ppo_res['ars']):.4f}")
    print(f"  Repair rate      mean={p_rr:.2%}")
    print(f"  Cap violations   total={p_cap_viol:.0f}  (repaired by best-fit)")
    print(f"  Conflict violations total={p_con_viol:.0f}  (repaired by best-fit)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  {'Method':<24} {'AR (mean±std)':<22} {'RepairRate':<10}")
    print(f"  {'-'*24} {'-'*22} {'-'*10}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} ± 0.0000       {'0':<10}  cap=0   conf=0")
    print(f"  {'PPO+Repair (P6)':<24} "
          f"{np.mean(ppo_res['ars']):.4f} ± {np.std(ppo_res['ars']):.4f}   "
          f"  {p_rr:<10.2%}  cap={p_cap_viol:.0f}  conf={p_con_viol:.0f}")
    print(f"{'='*62}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    log = {
        "scenario": sc_name,
        "prototype_scenario": prototype_name,
        "scenario_count": len(C.SCENARIOS),
        "train_count": len(C.TRAIN_SCENARIOS),
        "test_count": len(C.TEST_SCENARIOS),
        "N": N, "M": M,
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
        "ppo": {
            "ar_mean":                round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":                 round(float(np.std(ppo_res["ars"])),  6),
            "repair_rate_mean":       round(p_rr, 6),
            "repair_rate_std":        round(p_rr_std, 6),
            "cap_viol_total":         int(p_cap_viol),
            "conflict_viol_total":    int(p_con_viol),
            "placed_mean":            round(float(np.mean(ppo_res["placed"])), 2),
        },
        "training": {
            "total_steps":            C.TOTAL_STEPS,
            "n_episodes":             len(cb.episode_ars),
            "ar_last50":              round(float(np.mean(cb.episode_ars[-50:])), 6),
            "repair_rate_last50":     round(float(np.mean(cb.episode_repair_rates[-50:])), 6),
            "cap_viol_last50":        round(float(np.mean(cb.episode_cap_violations[-50:])), 4),
            "conflict_viol_last50":   round(float(np.mean(cb.episode_conflict_violations[-50:])), 4),
        }
    }
    log_path = C.OUTDIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" / "results.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = log_path.parent
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved → {log_path}")

    # ── Save CSV summary ───────────────────────────────────────────────────────
    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean",
                         "repair_rate", "cap_viol_total", "conflict_viol_total"])
        writer.writerow(["ILP (Optimal)", round(ilp_ar, 6), 0.0, M, 0.0, 0, 0])
        writer.writerow([
            "PPO+Repair (P6)",
            round(float(np.mean(ppo_res["ars"])), 6),
            round(float(np.std(ppo_res["ars"])), 6),
            round(float(np.mean(ppo_res["placed"])), 2),
            round(p_rr, 4),
            int(p_cap_viol),
            int(p_con_viol),
        ])
    print(f"  CSV  saved → {csv_path}")

    plot_training_curve(cb, ilp_ar, run_dir, sc_name)
    plot_comparison(ilp_ar, ppo_res, p_rr, p_rr_std, run_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {run_dir}/training_curve.png")
    print(f"  {run_dir}/comparison.png")
    print(f"  {run_dir}/results.json")
    print(f"  {run_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
