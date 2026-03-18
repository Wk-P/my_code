"""
evaluate.py — Evaluate the trained PPO agent and compare with random baseline.

P6 env: best-fit patch optimization.
Episodes always run to completion (M steps). Violations are recorded.

Run (after train.py):
    python problem6_ppo_opt/evaluate.py

Outputs saved to problem6_ppo_opt/results/:
    comparison_bar.png    — AR & violations bar chart (Random vs PPO)
    evaluation_log.json   — full numerical results
"""

import datetime
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from stable_baselines3 import PPO
import random
import config as C
from problem6_ppo_opt.env import P6Env
from problem2_ilp.objects import ECU, SVC


# ─────────────────────────────────────────────────────────────────────────────
#  Build environment
# ─────────────────────────────────────────────────────────────────────────────

def make_raw_env(seed: int = C.SEED) -> P6Env:
    random.seed(seed)
    caps, reqs = C.SCENARIOS[0]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return P6Env(ecus, services, scenarios=C.SCENARIOS)


# ─────────────────────────────────────────────────────────────────────────────
#  Run evaluation episodes
# ─────────────────────────────────────────────────────────────────────────────

def run_episodes(env: P6Env, policy, n_eps: int):
    """
    Run n_eps episodes. Episodes always complete (M steps, no early termination).
    Returns dict:
        ars              — final AR per episode
        total_violations — total violations per episode
        cap_violations   — capacity violations per episode
        dup_violations   — duplicate-ECU violations per episode
    """
    ars, total_viols, cap_viols, dup_viols = [], [], [], []

    for _ in range(n_eps):
        obs, _ = env.reset()
        done   = False
        info   = {}
        while not done:
            action = policy(obs)
            obs, reward, done, _, info = env.step(int(action))

        ars.append(info.get("ar", 0.0))
        total_viols.append(info.get("total_violations", 0))
        cap_viols.append(info.get("capacity_violations", 0))
        dup_viols.append(info.get("single_service_violations", 0))

    return {
        "ars":              np.array(ars),
        "total_violations": np.array(total_viols),
        "cap_violations":   np.array(cap_viols),
        "dup_violations":   np.array(dup_viols),
    }


def random_policy(obs):
    return np.random.randint(0, C.N)


# ─────────────────────────────────────────────────────────────────────────────
#  Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(random_res, ppo_res):
    def fmt(res):
        ars  = res["ars"]
        viol = res["total_violations"]
        return (f"{np.mean(ars):.4f} +/- {np.std(ars):.4f}",
                f"{np.mean(viol):.2f}")

    r_ar, r_viol = fmt(random_res)
    p_ar, p_viol = fmt(ppo_res)

    print(f"\n{'='*65}")
    print(f"  Evaluation Results  ({C.EVAL_EPS} episodes, N={C.N}, M={C.M})")
    print(f"  P6 = RL WITH best-fit patch optimization")
    print(f"{'='*65}")
    print(f"  {'Method':<24} {'AR (mean +/- std)':<24} {'Viol/ep'}")
    print(f"  {'-'*24} {'-'*24} {'-'*10}")
    print(f"  {'Random Baseline':<24} {r_ar:<24} {r_viol}")
    print(f"  {'PPO (P6, best-fit)':<24} {p_ar:<24} {p_viol}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(random_res, ppo_res, outdir: Path):
    labels  = ["Random\nBaseline", "PPO\n(no constraint)"]
    colors  = ["steelblue", "coral"]

    r_ar = random_res["ars"]
    p_ar = ppo_res["ars"]
    r_viol = random_res["total_violations"]
    p_viol = ppo_res["total_violations"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"P6 Evaluation  |  N={C.N}  M={C.M}  ({C.EVAL_EPS} episodes)", fontsize=13)

    # ── Left: AR box plot ─────────────────────────────────────────────────────
    bp = ax1.boxplot(
        [r_ar, p_ar],
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for i, data in enumerate([r_ar, p_ar]):
        mean_v = np.mean(data)
        ax1.text(i + 1, mean_v + 0.03, f"mu={mean_v:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax1.set_title("AR Distribution", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # ── Right: Violation count bar ────────────────────────────────────────────
    vr_means = [np.mean(r_viol), np.mean(p_viol)]
    vr_stds  = [np.std(r_viol),  np.std(p_viol)]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75,
                   yerr=vr_stds, capsize=6, ecolor="black")
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")
    ax2.set_ylabel("Avg Constraint Violations per Episode", fontsize=11)
    ax2.set_title("Constraint Violations", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)

    model_zip = Path(str(C.MODEL_PATH) + ".zip")
    if not model_zip.exists():
        print(f"[ERROR] Model not found: {model_zip}")
        print("  Run train.py first.")
        return

    print(f"\nLoading model from {model_zip} ...")
    model = PPO.load(str(C.MODEL_PATH))

    env = make_raw_env(seed=C.SEED)

    # ── Random baseline ───────────────────────────────────────────────────────
    print(f"Running random baseline ({C.EVAL_EPS} eps) ...")
    np.random.seed(C.SEED)
    random_res = run_episodes(env, lambda obs: random_policy(obs), C.EVAL_EPS)

    # ── PPO agent ─────────────────────────────────────────────────────────────
    print(f"Running PPO agent ({C.EVAL_EPS} eps, deterministic) ...")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ppo_res = run_episodes(env, ppo_policy, C.EVAL_EPS)

    print_table(random_res, ppo_res)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    log = {
        "N": C.N, "M": C.M, "eval_eps": C.EVAL_EPS,
        "random": {
            "ar_mean":   round(float(np.mean(random_res["ars"])), 6),
            "ar_std":    round(float(np.std(random_res["ars"])), 6),
            "viol_mean": round(float(np.mean(random_res["total_violations"])), 4),
            "cap_viol_mean": round(float(np.mean(random_res["cap_violations"])), 4),
            "dup_viol_mean": round(float(np.mean(random_res["dup_violations"])), 4),
        },
        "ppo": {
            "ar_mean":   round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":    round(float(np.std(ppo_res["ars"])), 6),
            "viol_mean": round(float(np.mean(ppo_res["total_violations"])), 4),
            "cap_viol_mean": round(float(np.mean(ppo_res["cap_violations"])), 4),
            "dup_viol_mean": round(float(np.mean(ppo_res["dup_violations"])), 4),
        },
    }
    log_path = C.OUTDIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" / "evaluation_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = log_path.parent
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved -> {log_path}")

    plot_comparison(random_res, ppo_res, run_dir)
    print("Evaluation complete.\n")


if __name__ == "__main__":
    pass
    #  main()