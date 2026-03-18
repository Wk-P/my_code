"""
evaluate.py — Evaluate Lagrangian PPO vs Random (no mask).

Run (after train.py or run.py):
    python problem5_lagarange/evaluate.py
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

from stable_baselines3 import PPO
import random
import config as C
from problem5_lagrangian.env import LagrangeEnv
from problem2_single.objects import ECU, SVC


def make_raw_env(seed: int = C.SEED) -> LagrangeEnv:
    random.seed(seed)
    caps, reqs = C.SCENARIOS[C.SCENARIO_IDX]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return LagrangeEnv(ecus, services, scenarios=C.SCENARIOS, lambda_init=0.0)


def run_episodes(env: LagrangeEnv, policy_fn, n_eps: int):
    """policy_fn(obs) -> int"""
    ars, viol_rates, placed = [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action = policy_fn(obs)
            obs, _, done, _, info = env.step(int(action))
        ars.append(info.get("ar", 0.0))
        viol_rates.append(info.get("viol_rate_ep", 0.0))
        placed.append(info.get("services_placed", 0))
    return {
        "ars":        np.array(ars),
        "viol_rates": np.array(viol_rates),
        "placed":     np.array(placed),
    }


def random_policy(obs):
    return np.random.randint(0, C.N)


def print_table(rand_res, ppo_res):
    def fmt(res):
        return (
            f"{np.mean(res['ars']):.4f} ± {np.std(res['ars']):.4f}",
            f"{np.mean(res['viol_rates']):.2%}",
            f"{np.mean(res['placed']):.1f}/{C.M}",
        )
    r_ar, r_vr, r_pl = fmt(rand_res)
    p_ar, p_vr, p_pl = fmt(ppo_res)

    print(f"\n{'='*72}")
    print(f"  P5 Evaluation  ({C.EVAL_EPS} episodes, N={C.N}, M={C.M})")
    print(f"  Lagrangian relaxation — NO action masking")
    print(f"{'='*72}")
    print(f"  {'Method':<22} {'AR (mean ± std)':<26} {'Viol%':<14} {'Placed'}")
    print(f"  {'-'*22} {'-'*26} {'-'*14} {'-'*8}")
    print(f"  {'Random (no mask)':<22} {r_ar:<26} {r_vr:<14} {r_pl}")
    print(f"  {'Lagrange PPO':<22} {p_ar:<26} {p_vr:<14} {p_pl}")
    print(f"{'='*72}\n")


def plot_comparison(rand_res, ppo_res, outdir: Path):
    labels = ["Random\n(no mask)", "Lagrange\nPPO"]
    colors = ["steelblue", "darkorange"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"P5 Lagrangian Evaluation  |  N={C.N}  M={C.M}  ({C.EVAL_EPS} eps)",
        fontsize=13,
    )

    # AR box plot
    bp = ax1.boxplot(
        [rand_res["ars"], ppo_res["ars"]],
        tick_labels=labels, patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    for i, data in enumerate([rand_res["ars"], ppo_res["ars"]]):
        mv = np.mean(data)
        ax1.text(i+1, mv+0.02, f"mu={mv:.3f}", ha="center", fontsize=9,
                 fontweight="bold", color=colors[i])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax1.set_title("AR Distribution", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Violation rate bar
    vr_means = [np.mean(rand_res["viol_rates"]), np.mean(ppo_res["viol_rates"])]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75)
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.01,
                 f"{v:.2%}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Violation Rate (per step, per episode)", fontsize=11)
    ax2.set_title("Constraint Violations", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison_eval.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)

    model_zip = Path(str(C.MODEL_PATH) + ".zip")
    if not model_zip.exists():
        print(f"[ERROR] Model not found: {model_zip}")
        print("  Run train.py or run.py first.")
        return

    print(f"\nLoading model from {model_zip} ...")
    model = PPO.load(str(C.MODEL_PATH))

    env = make_raw_env(seed=C.SEED)

    print(f"Running random baseline ({C.EVAL_EPS} eps, no masking) ...")
    np.random.seed(C.SEED)
    rand_res = run_episodes(env, random_policy, C.EVAL_EPS)

    print(f"Running Lagrange PPO ({C.EVAL_EPS} eps, deterministic) ...")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ppo_res = run_episodes(env, ppo_policy, C.EVAL_EPS)

    print_table(rand_res, ppo_res)

    log = {
        "N": C.N, "M": C.M, "eval_eps": C.EVAL_EPS,
        "random": {
            "ar_mean":        round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":         round(float(np.std(rand_res["ars"])), 6),
            "viol_rate_mean": round(float(np.mean(rand_res["viol_rates"])), 6),
        },
        "lagrange_ppo": {
            "ar_mean":        round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":         round(float(np.std(ppo_res["ars"])), 6),
            "viol_rate_mean": round(float(np.mean(ppo_res["viol_rates"])), 6),
        },
    }
    run_dir = C.OUTDIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "eval_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved -> {run_dir / 'eval_log.json'}")

    plot_comparison(rand_res, ppo_res, run_dir)
    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()
