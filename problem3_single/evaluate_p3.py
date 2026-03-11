"""
evaluate_p3.py — Evaluate the trained PPO agent and compare with random baseline.

Run (after train_p3.py):
    python problem3_single/evaluate_p3.py

Outputs saved to problem3_single/results/:
    comparison_bar.png    — AR & violations bar chart (Random vs PPO)
    evaluation_log.json   — full numerical results
"""

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
from env_p3 import P3Env
from problem2_single.objects import ECU, SVC


# ─────────────────────────────────────────────────────────────────────────────
#  Build the same fixed environment used during training
# ─────────────────────────────────────────────────────────────────────────────

def make_raw_env(seed: int = C.SEED) -> P3Env:
    """Return an unwrapped P3Env (no Monitor) that randomly samples
    from all 200 scenarios on each reset() call."""
    random.seed(seed)
    caps, reqs = C.SCENARIOS[0]   # 初始 scenario，reset() 后随机替换
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    return P3Env(ecus, services, scenarios=C.SCENARIOS)


# ─────────────────────────────────────────────────────────────────────────────
#  Run evaluation episodes
# ─────────────────────────────────────────────────────────────────────────────

def run_episodes(env: P3Env, policy, n_eps: int, deterministic: bool = True):
    """
    Run n_eps episodes with the given policy callable.
    policy(obs) -> action

    Returns dict of arrays:
        ars        — final AR (0.0 if episode terminated early due to violation)
        completed  — bool, whether episode finished without any violation
    """
    ars, completed = [], []

    for _ in range(n_eps):
        obs, _ = env.reset()
        done   = False
        info   = {"ar": 0.0}
        success = True

        while not done:
            action = policy(obs)
            obs, reward, done, _, info = env.step(int(action))
            if reward == -1.0:   # violation → terminated early
                success = False

        ars.append(info["ar"])
        completed.append(success)

    return {
        "ars":       np.array(ars),
        "completed": np.array(completed),
    }


def random_policy(obs):
    """Uniformly random action."""
    return np.random.randint(0, C.N)


# ─────────────────────────────────────────────────────────────────────────────
#  Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(random_res, ppo_res):
    def fmt(res):
        ars  = res["ars"]
        comp = res["completed"]
        comp_rate = np.mean(comp) * 100
        ar_mean = np.mean(ars[comp]) if comp.any() else 0.0
        ar_std  = np.std(ars[comp])  if comp.any() else 0.0
        return (f"{ar_mean:.4f} ± {ar_std:.4f}",
                f"{comp_rate:.1f}%")

    r_ar, r_cr = fmt(random_res)
    p_ar, p_cr = fmt(ppo_res)

    print(f"\n{'='*60}")
    print(f"  Evaluation Results  ({C.EVAL_EPS} episodes each,  N={C.N}  M={C.M})")
    print(f"{'='*60}")
    print(f"  {'Method':<24} {'AR (mean ± std)':<22} {'Completion Rate'}")
    print(f"  {'-'*24} {'-'*22} {'-'*15}")
    print(f"  {'Random Baseline':<24} {r_ar:<22} {r_cr}")
    print(f"  {'PPO (P3)':<24} {p_ar:<22} {p_cr}")
    print(f"{'='*60}")

    p_comp = np.mean(ppo_res["completed"])
    r_comp = np.mean(random_res["completed"])
    if p_comp > 0 and r_comp > 0:
        gain = np.mean(ppo_res["ars"][ppo_res["completed"]]) - np.mean(random_res["ars"][random_res["completed"]])
        print(f"  AR gain (PPO vs Random, completed only): {gain:+.4f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(random_res, ppo_res, outdir: Path):
    labels  = ["Random\nBaseline", "PPO\n(no constraint)"]
    colors  = ["steelblue", "coral"]

    r_ar = random_res["ars"]
    p_ar = ppo_res["ars"]
    r_vr = random_res["total_violations"] / C.M
    p_vr = ppo_res["total_violations"] / C.M

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"P3 Evaluation  |  N={C.N}  M={C.M}  ({C.EVAL_EPS} episodes)", fontsize=13)

    # ── Left: AR box plot ─────────────────────────────────────────────────────
    bp = ax1.boxplot(
        [r_ar, p_ar],
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, (data, label) in enumerate(zip([r_ar, p_ar], labels)):
        mean_v = np.mean(data)
        ax1.text(i + 1, mean_v + 0.03, f"μ={mean_v:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=colors[i])

    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax1.set_title("AR Distribution", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # ── Right: Violation rate bar ─────────────────────────────────────────────
    vr_means = [np.mean(r_vr), np.mean(p_vr)]
    vr_stds  = [np.std(r_vr),  np.std(p_vr)]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75,
                   yerr=vr_stds, capsize=6, ecolor="black")

    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylim(0, max(vr_means) * 1.5 + 0.05)
    ax2.set_ylabel("Constraint Violation Rate (violations / M)", fontsize=11)
    ax2.set_title("Constraint Violation Rate", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    C.OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Load trained model ────────────────────────────────────────────────────
    model_zip = Path(str(C.MODEL_PATH) + ".zip")
    if not model_zip.exists():
        print(f"[ERROR] Model not found: {model_zip}")
        print("  Run train_p3.py first.")
        return

    print(f"\nLoading model from {model_zip} …")
    model = PPO.load(str(C.MODEL_PATH))

    # ── Shared environment (same problem instance as training) ────────────────
    env = make_raw_env(seed=C.SEED)

    # ── Random baseline ───────────────────────────────────────────────────────
    print(f"Running random baseline ({C.EVAL_EPS} eps) …")
    np.random.seed(C.SEED)
    random_res = run_episodes(env, lambda obs: random_policy(obs), C.EVAL_EPS)

    # ── PPO agent (deterministic) ─────────────────────────────────────────────
    print(f"Running PPO agent ({C.EVAL_EPS} eps, deterministic) …")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    ppo_res = run_episodes(env, ppo_policy, C.EVAL_EPS)

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(random_res, ppo_res)

    # ── Save JSON log ─────────────────────────────────────────────────────────
    log = {
        "N": C.N, "M": C.M, "eval_eps": C.EVAL_EPS,
        "random": {
            "completion_rate":   round(float(np.mean(random_res["completed"])), 6),
            "ar_mean":           round(float(np.mean(random_res["ars"][random_res["completed"]])) if random_res["completed"].any() else 0.0, 6),
            "ar_std":            round(float(np.std(random_res["ars"][random_res["completed"]]))  if random_res["completed"].any() else 0.0, 6),
        },
        "ppo": {
            "completion_rate":   round(float(np.mean(ppo_res["completed"])), 6),
            "ar_mean":           round(float(np.mean(ppo_res["ars"][ppo_res["completed"]])) if ppo_res["completed"].any() else 0.0, 6),
            "ar_std":            round(float(np.std(ppo_res["ars"][ppo_res["completed"]]))  if ppo_res["completed"].any() else 0.0, 6),
        },
    }
    log_path = C.OUTDIR / "evaluation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved → {log_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(random_res, ppo_res, C.OUTDIR)

    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()
