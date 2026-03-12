"""
evaluate_p4.py — Evaluate MaskablePPO vs Random (masked) vs ILP.

Run (after train_p4.py):
    python problem4_single/evaluate_p4.py
"""

import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from sb3_contrib import MaskablePPO
import random
import config as C
from env_p4 import P4Env
from problem2_single.objects import ECU, SVC


def make_raw_env(seed: int = C.SEED) -> P4Env:
    random.seed(seed)
    caps, reqs = C.SCENARIOS[0]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return P4Env(ecus, services, scenarios=C.SCENARIOS)


def run_episodes(env: P4Env, policy_fn, n_eps: int):
    """
    policy_fn(obs, mask) -> int
    Returns dict: ars, services_placed
    """
    ars, placed = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            mask = env.action_masks()
            if not np.any(mask):
                break
            action = policy_fn(obs, mask)
            obs, _, done, _, info = env.step(int(action))
        ars.append(info.get("ar", 0.0))
        placed.append(info.get("services_placed", 0))
    return {
        "ars":    np.array(ars),
        "placed": np.array(placed),
    }


def random_masked_policy(obs, mask):
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return 0
    return np.random.choice(valid)


def print_table(rand_res, ppo_res):
    def fmt(res):
        ars = res["ars"]
        pl  = res["placed"]
        return (f"{np.mean(ars):.4f} +/- {np.std(ars):.4f}",
                f"{np.mean(pl):.1f}/{C.M}")

    r_ar, r_pl = fmt(rand_res)
    p_ar, p_pl = fmt(ppo_res)

    print(f"\n{'='*70}")
    print(f"  P4 Evaluation  ({C.EVAL_EPS} episodes, N={C.N}, M={C.M})")
    print(f"  Action masking → 0 constraint violations guaranteed")
    print(f"{'='*70}")
    print(f"  {'Method':<28} {'AR (mean +/- std)':<24} {'Placed'}")
    print(f"  {'-'*28} {'-'*24} {'-'*10}")
    print(f"  {'Random (masked)':<28} {r_ar:<24} {r_pl}")
    print(f"  {'MaskablePPO (P4)':<28} {p_ar:<24} {p_pl}")
    print(f"{'='*70}\n")


def plot_comparison(rand_res, ppo_res, outdir: Path):
    labels = ["Random\n(masked)", "MaskablePPO\n(P4)"]
    colors = ["steelblue", "seagreen"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"P4 Evaluation  |  N={C.N}  M={C.M}  ({C.EVAL_EPS} episodes)", fontsize=13)

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
        ax1.text(i+1, mv+0.03, f"mu={mv:.3f}", ha="center", fontsize=9,
                 fontweight="bold", color=colors[i])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax1.set_title("AR Distribution (0 violations)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Services placed bar
    pl_means = [np.mean(rand_res["placed"]), np.mean(ppo_res["placed"])]
    pl_stds  = [np.std(rand_res["placed"]),  np.std(ppo_res["placed"])]
    bars = ax2.bar(labels, pl_means, color=colors, alpha=0.75,
                   yerr=pl_stds, capsize=6, ecolor="black")
    for bar, v in zip(bars, pl_means):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.1,
                 f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, C.M + 1)
    ax2.axhline(C.M, color="red", linestyle="--", alpha=0.5, label=f"M={C.M}")
    ax2.set_ylabel("Services Placed per Episode", fontsize=11)
    ax2.set_title("Placement Success", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)

    model_zip = Path(str(C.MODEL_PATH) + ".zip")
    if not model_zip.exists():
        print(f"[ERROR] Model not found: {model_zip}")
        print("  Run train_p4.py first.")
        return

    print(f"\nLoading model from {model_zip} ...")
    model = MaskablePPO.load(str(C.MODEL_PATH))
    env = make_raw_env(seed=C.SEED)

    print(f"Running random masked baseline ({C.EVAL_EPS} eps) ...")
    np.random.seed(C.SEED)
    rand_res = run_episodes(env, random_masked_policy, C.EVAL_EPS)

    print(f"Running MaskablePPO ({C.EVAL_EPS} eps, deterministic) ...")
    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return int(action)
    ppo_res = run_episodes(env, ppo_policy, C.EVAL_EPS)

    print_table(rand_res, ppo_res)

    log = {
        "N": C.N, "M": C.M, "eval_eps": C.EVAL_EPS,
        "random_masked": {
            "ar_mean":   round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":    round(float(np.std(rand_res["ars"])), 6),
            "placed_mean": round(float(np.mean(rand_res["placed"])), 2),
        },
        "maskable_ppo": {
            "ar_mean":   round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":    round(float(np.std(ppo_res["ars"])), 6),
            "placed_mean": round(float(np.mean(ppo_res["placed"])), 2),
        },
    }
    log_path = C.OUTDIR / "evaluation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved -> {log_path}")

    plot_comparison(rand_res, ppo_res, C.OUTDIR)
    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()
