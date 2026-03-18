"""
evaluate.py — Evaluate DQN vs Random baseline (NO action masking).

Run (after train.py or run_all.py):
    python dqn/evaluate.py
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
sys.path.insert(0, str(HERE.parent))

from stable_baselines3 import DQN
import random
import config as C
from dqn.env import DQNEnv
from problem2_ilp.objects import ECU, SVC


def make_raw_env(seed: int = C.SEED) -> DQNEnv:
    random.seed(seed)
    caps, reqs = C.SCENARIOS[C.SCENARIO_IDX]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return DQNEnv(ecus, services, scenarios=C.SCENARIOS)


def run_episodes(env: DQNEnv, policy_fn, n_eps: int): 
    """
    policy_fn(obs) -> int   (no mask)
    Returns dict: ars, placed, viols
    """
    ars, placed, viols = [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action = policy_fn(obs)
            obs, _, done, _, info = env.step(int(action))
        ars.append(info.get("ar", 0.0))
        placed.append(info.get("services_placed", 0))
        viols.append(1 if info.get("violated", False) else 0)
    return {
        "ars":    np.array(ars),
        "placed": np.array(placed),
        "viols":  np.array(viols),
    }


def random_policy(obs):
    return np.random.randint(0, C.N)


def print_table(rand_res, dqn_res):
    def fmt(res):
        return (f"{np.mean(res['ars']):.4f} +/- {np.std(res['ars']):.4f}",
                f"{np.mean(res['placed']):.1f}/{C.M}",
                f"{np.mean(res['viols']):.2%}")

    r_ar, r_pl, r_vr = fmt(rand_res)
    d_ar, d_pl, d_vr = fmt(dqn_res)

    print(f"\n{'='*72}")
    print(f"  DQN Evaluation  ({C.EVAL_EPS} episodes, N={C.N}, M={C.M})")
    print(f"  NO action masking \u2014 violations terminate the episode.")
    print(f"{'='*72}")
    print(f"  {'Method':<24} {'AR (mean +/- std)':<24} {'Placed':<12} {'Viol%'}")
    print(f"  {'-'*24} {'-'*24} {'-'*12} {'-'*6}")
    print(f"  {'Random (no mask)':<24} {r_ar:<24} {r_pl:<12} {r_vr}")
    print(f"  {'DQN':<24} {d_ar:<24} {d_pl:<12} {d_vr}")
    print(f"{'='*72}\n")


def plot_comparison(rand_res, dqn_res, outdir: Path):
    labels = ["Random\n(no mask)", "DQN\n(no mask)"]
    colors = ["steelblue", "seagreen"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"DQN Evaluation  |  N={C.N}  M={C.M}  ({C.EVAL_EPS} episodes)", fontsize=13)

    # AR box plot
    bp = ax1.boxplot(
        [rand_res["ars"], dqn_res["ars"]],
        tick_labels=labels, patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    for i, data in enumerate([rand_res["ars"], dqn_res["ars"]]):
        mv = np.mean(data)
        ax1.text(i+1, mv+0.03, f"mu={mv:.3f}", ha="center", fontsize=9,
                 fontweight="bold", color="black")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax1.set_title("AR Distribution (valid episodes only)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Violation rate bar
    vr_means = [np.mean(rand_res["viols"]), np.mean(dqn_res["viols"])]
    vr_stds  = [np.std(rand_res["viols"]),  np.std(dqn_res["viols"])]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75,
                   yerr=vr_stds, capsize=6, ecolor="black")
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.02,
                 f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color="black")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Violation Rate per Episode", fontsize=11)
    ax2.set_title("Constraint Violations", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison_eval.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)

    # Find the latest dqn_model.zip
    model_files = sorted(C.OUTDIR.rglob("dqn_model.zip"))
    if not model_files:
        print(f"[ERROR] No dqn_model.zip found under {C.OUTDIR}")
        print("  Run train.py or run_all.py first.")
        return
    model_zip = model_files[-1]
    model_dir = model_zip.parent
    print(f"\nLoading model from {model_zip} ...")
    model = DQN.load(str(model_zip.with_suffix("")))

    env = make_raw_env(seed=C.SEED)

    print(f"Running random baseline ({C.EVAL_EPS} eps, no masking) ...")
    np.random.seed(C.SEED)
    rand_res = run_episodes(env, random_policy, C.EVAL_EPS)

    print(f"Running DQN ({C.EVAL_EPS} eps, deterministic) ...")
    def dqn_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    dqn_res = run_episodes(env, dqn_policy, C.EVAL_EPS)

    print_table(rand_res, dqn_res)

    log = {
        "N": C.N, "M": C.M, "eval_eps": C.EVAL_EPS,
        "random": {
            "ar_mean":     round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":      round(float(np.std(rand_res["ars"])), 6),
            "placed_mean": round(float(np.mean(rand_res["placed"])), 2),
            "viol_rate":   round(float(np.mean(rand_res["viols"])), 4),
        },
        "dqn": {
            "ar_mean":     round(float(np.mean(dqn_res["ars"])), 6),
            "ar_std":      round(float(np.std(dqn_res["ars"])), 6),
            "placed_mean": round(float(np.mean(dqn_res["placed"])), 2),
            "viol_rate":   round(float(np.mean(dqn_res["viols"])), 4),
        },
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = model_dir / f"evaluation_{ts}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved -> {log_path}")

    plot_comparison(rand_res, dqn_res, model_dir)
    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()
