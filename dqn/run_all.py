"""
run_all.py — One-shot DQN full pipeline:

  1. Load scenario from YAML
  2. Solve with ILP (PuLP)             -> ilp_ar (optimal upper bound)
  3. Evaluate random policy (no mask)  -> random_ars + violations
  4. Train DQN (no action masking)     -> training curve
  5. Evaluate trained DQN              -> dqn_ars + violations
  6. Produce plots:
       - comparison.png     — AR box plot + violation rate bar (3-way)
       - training_curve.png — reward & violation rate during training

DQN Design: NO action masking.
  - Constraint violations → reward = -1, episode terminates immediately.
  - Agent learns to avoid violations through reward shaping.

Run:
    python dqn/run_all.py
"""

import datetime
import csv
import sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch
import yaml
import pulp
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import config as C
from dqn.env import DQNEnv
from problem2_single.objects import ECU, SVC


def resolve_device(cfg: str) -> str:
    if cfg != "auto":
        return cfg
    if torch.cuda.is_available():
        print(f"[CUDA] {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[CPU] No CUDA GPU, using CPU")
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Load scenario
# ══════════════════════════════════════════════════════════════════════════════

def load_scenario():
    with open(C.YAML_CONFIG, "r") as f:
        data = yaml.safe_load(f)
    scenario = data["scenarios"][C.SCENARIO_IDX]
    ecus     = [ECU(s["name"], s["capacity"])    for s in scenario["ECUs"]]
    services = [SVC(s["name"], s["requirement"]) for s in scenario["SVCs"]]
    N, M = len(ecus), len(services)
    print(f"Loaded: {scenario['name']}  |  N={N} ECUs  M={M} SVCs")
    print(f"  ECU capacities : {[e.capacity for e in ecus]}")
    print(f"  SVC requirements: {[s.requirement for s in services]}")
    return ecus, services, scenario["name"]


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — ILP optimal
# ══════════════════════════════════════════════════════════════════════════════

def solve_ilp(ecus, services):
    N, M = len(ecus), len(services)
    e_list = [e.capacity    for e in ecus]
    n_list = [s.requirement for s in services]

    prob = pulp.LpProblem("DQN_ILP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(M), range(N)), cat="Binary")

    prob += pulp.lpSum(x[i][j] * n_list[i] / e_list[j]
                       for i in range(M) for j in range(N))
    for i in range(M):
        prob += pulp.lpSum(x[i][j] for j in range(N)) == 1
    for j in range(N):
        prob += pulp.lpSum(x[i][j] for i in range(M)) <= 1
    for i in range(M):
        for j in range(N):
            if n_list[i] > e_list[j]:
                prob += x[i][j] == 0
    for j in range(N):
        prob += pulp.lpSum(x[i][j] * n_list[i] for i in range(M)) <= e_list[j]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    alloc = {}
    for j in range(N):
        svcs = [i for i in range(M) if pulp.value(x[i][j]) == 1]
        if svcs:
            alloc[j] = {
                "services":    svcs,
                "utilization": sum(n_list[i] for i in svcs) / e_list[j],
                "capacity":    e_list[j],
                "demand":      sum(n_list[i] for i in svcs),
            }
    total_util  = pulp.value(prob.objective) or 0.0
    active_ecus = len(alloc)
    avg_util    = total_util / active_ecus if active_ecus > 0 else 0.0
    return {
        "status":            pulp.LpStatus[prob.status],
        "avg_utilization":   avg_util,
        "total_utilization": total_util,
        "active_ecus":       active_ecus,
        "allocation":        alloc,
    }


def solve_ilp_all_scenarios():
    """Return (mean_ar, per_scenario_results) for all scenarios.
    Priority: 1) shared p2 cache  2) own local cache  3) compute from scratch.
    """
    cache_key = f"{C.YAML_CONFIG.name}__n{len(C.SCENARIOS)}"

    # ─ 1. Shared cache written by problem2_single/optimal_solution/main.py ─
    shared_cache = C.YAML_CONFIG.parent.parent / "results" / "ilp_cache.json"
    if shared_cache.exists():
        with open(shared_cache) as f:
            cache = json.load(f)
        if cache.get("key") == cache_key and len(cache.get("results", [])) == len(C.SCENARIOS):
            print(f"    [cache] Loaded ILP results from shared p2 cache")
            results = cache["results"]
            ars = [r["avg_utilization"] for r in results]
            return float(np.mean(ars)), results

    # ─ 2. Own local incremental cache ─
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    cache_path = C.OUTDIR / "ilp_cache.json"
    results: list = []
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get("key") == cache_key:
            results = cache.get("results", [])
            if len(results) == len(C.SCENARIOS):
                print(f"    [cache] Loaded ILP results from {cache_path}")
                ars = [r["avg_utilization"] for r in results]
                return float(np.mean(ars)), results
            print(f"    [cache] Resuming from scenario {len(results)+1}")

    # ─ 3. Compute remaining, save after each ─
    for idx, (caps, reqs) in enumerate(C.SCENARIOS[len(results):], start=len(results)):
        ecus_sc = [ECU(f"ECU{i}", c) for i, c in enumerate(caps)]
        svcs_sc = [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)]
        res = solve_ilp(ecus_sc, svcs_sc)
        results.append(res)
        print(f"    Scenario {idx+1}: AR={res['avg_utilization']:.4f}  ({res['status']})")
        with open(cache_path, "w") as f:
            json.dump({"key": cache_key, "results": results}, f)

    print(f"    [cache] Saved to {cache_path}")
    ars = [r["avg_utilization"] for r in results]
    return float(np.mean(ars)), results


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 & 5 — Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps):
    """policy_fn(obs) -> int   (no mask)"""
    env = DQNEnv(ecus, services, scenarios=C.SCENARIOS)
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

class DQNCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards:  list[float] = []
        self.episode_placed:   list[int]   = []
        self.episode_violated: list[int]   = []
        self.timesteps_at_ep:  list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_placed.append(int(info.get("services_placed", 0)))
                self.episode_violated.append(1 if info.get("violated", False) else 0)
                self.timesteps_at_ep.append(self.num_timesteps)
        return True


def train_dqn(ecus, services):
    caps = [e.capacity    for e in ecus]
    reqs = [s.requirement for s in services]
    env  = Monitor(DQNEnv(
        [ECU(f"ECU{i}", c) for i, c in enumerate(caps)],
        [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)],
        scenarios=C.SCENARIOS,
    ))
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
        device                 = resolve_device(C.DEVICE),
        verbose                = 0,
        seed                   = C.SEED,
    )
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep     = len(cb.episode_rewards)
    last50_r = np.mean(cb.episode_rewards[-50:]) if n_ep >= 50 else np.mean(cb.episode_rewards)
    last50_v = np.mean(cb.episode_violated[-50:]) if n_ep >= 50 else np.mean(cb.episode_violated)
    print(f"  Training done  {elapsed:.1f}s | {n_ep} eps "
          f"| reward(last50)={last50_r:.4f} | viol_rate(last50)={last50_v:.2%}")
    return model, cb


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def moving_avg(arr, w):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    return np.convolve(arr, np.ones(w) / w, mode="valid"), w - 1


def plot_training_curve(cb, ilp_ar, outdir, scenario_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ts = np.array(cb.timesteps_at_ep)

    sm, off = moving_avg(cb.episode_rewards, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_rewards, color="seagreen", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off:off+len(sm)], sm, color="seagreen", linewidth=2,
             label=f"DQN reward (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.set_ylim(-1.1, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"DQN Training \u2014 {scenario_name}  ({C.TOTAL_STEPS:,} steps)", fontsize=12)
    ax1.grid(alpha=0.3)

    viol_rate = np.array(cb.episode_violated, dtype=float)
    sm_v, off_v = moving_avg(viol_rate, C.SMOOTH_W)
    ax2.plot(ts, viol_rate, color="tomato", alpha=0.2, linewidth=0.8)
    ax2.plot(ts[off_v:off_v+len(sm_v)], sm_v, color="tomato", linewidth=2,
             label="violation rate (smoothed)")
    ax2.set_ylabel("Violation Rate", fontsize=11)
    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_comparison(ilp_ar, rand_res, dqn_res, outdir, scenario_name):
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels = ["ILP\n(Optimal)", "Random\n(no mask)", "DQN\n(no mask)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ILP vs Random vs DQN \u2014 {scenario_name}", fontsize=13, fontweight="bold")

    # AR box plot
    ax = axes[0]
    bp = ax.boxplot(
        [rand_res["ars"], dqn_res["ars"]],
        positions=[2, 3], widths=0.5, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors[1:]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.axhline(ilp_ar, color=colors[0], linestyle="--", linewidth=2, alpha=0.9,
               label=f"ILP  AR={ilp_ar:.4f}")
    ax.plot(1, ilp_ar, marker="D", color=colors[0], markersize=10, zorder=5)

    for pos, data, color in zip([2, 3], [rand_res["ars"], dqn_res["ars"]], colors[1:]):
        mv = np.mean(data)
        ax.text(pos, mv + 0.02, f"mu={mv:.3f}", ha="center", fontsize=9,
                fontweight="bold", color=color)

    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax.set_title("AR Distribution (valid episodes only)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Violation rate bar
    ax2 = axes[1]
    vr_means = [0.0, np.mean(rand_res["viols"]), np.mean(dqn_res["viols"])]
    vr_stds  = [0.0, np.std(rand_res["viols"]),  np.std(dqn_res["viols"])]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75,
                   yerr=vr_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Violation Rate per Episode", fontsize=11)
    ax2.set_title("Constraint Violations", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    base_dir = C.OUTDIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DQN run_all.py  \u2014  RL WITHOUT action masking")
    print(f"  Violation \u2192 reward=-1, episode terminates immediately.")
    print(f"  Config : {C.YAML_CONFIG.name}  Scenario idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")

    # 1. Load scenario
    ecus, services, sc_name = load_scenario()
    N, M = len(ecus), len(services)

    # 2. ILP (all scenarios)
    print(f"\n[1/4] Solving ILP for all {len(C.SCENARIOS)} scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios()
    print(f"  ILP mean AR across {len(C.SCENARIOS)} scenarios: {ilp_ar:.4f}")

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

    # 4. DQN training
    print(f"\n[3/4] DQN training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_dqn(ecus, services)
    model_path = base_dir / "dqn_model"
    model.save(str(model_path))
    print(f"  Model saved -> {model_path}.zip")

    # 5. DQN evaluation
    print(f"\n[4/4] DQN evaluation ({C.EVAL_EPS} episodes, deterministic) ...")
    def dqn_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    dqn_res = run_episodes(ecus, services, dqn_policy, C.EVAL_EPS)
    print(f"  DQN AR  mean={np.mean(dqn_res['ars']):.4f}  "
          f"std={np.std(dqn_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(dqn_res['placed']):.1f}/{M}")
    print(f"  Viol rate  {np.mean(dqn_res['viols']):.2%}")

    # Summary
    print(f"\n{'='*68}")
    print(f"  {'Method':<24} {'AR (mean+/-std)':<22} {'Placed':<10} {'Viol%'}")
    print(f"  {'-'*24} {'-'*22} {'-'*10} {'-'*6}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} +/- 0.0000     {M}/{M:<6} 0%")
    print(f"  {'Random (no mask)':<24} "
          f"{np.mean(rand_res['ars']):.4f} +/- {np.std(rand_res['ars']):.4f}   "
          f"  {np.mean(rand_res['placed']):.1f}/{M:<2}   {np.mean(rand_res['viols']):.0%}")
    print(f"  {'DQN (no mask)':<24} "
          f"{np.mean(dqn_res['ars']):.4f} +/- {np.std(dqn_res['ars']):.4f}   "
          f"  {np.mean(dqn_res['placed']):.1f}/{M:<2}   {np.mean(dqn_res['viols']):.0%}")
    print(f"{'='*68}\n")

    # Save JSON
    log = {
        "scenario": sc_name, "N": N, "M": M,
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
        "dqn": {
            "ar_mean":     round(float(np.mean(dqn_res["ars"])), 6),
            "ar_std":      round(float(np.std(dqn_res["ars"])), 6),
            "placed_mean": round(float(np.mean(dqn_res["placed"])), 2),
            "viol_rate":   round(float(np.mean(dqn_res["viols"])), 4),
        },
        "training": {
            "total_steps":      C.TOTAL_STEPS,
            "n_episodes":       len(cb.episode_rewards),
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
        writer.writerow(["ILP (Optimal)",    round(ilp_ar, 6), 0.0, M, 0.0])
        writer.writerow(["Random (no mask)", round(float(np.mean(rand_res["ars"])), 6),
                         round(float(np.std(rand_res["ars"])), 6),
                         round(float(np.mean(rand_res["placed"])), 2),
                         round(float(np.mean(rand_res["viols"])), 4)])
        writer.writerow(["DQN (no mask)",    round(float(np.mean(dqn_res["ars"])),  6),
                         round(float(np.std(dqn_res["ars"])),  6),
                         round(float(np.mean(dqn_res["placed"])), 2),
                         round(float(np.mean(dqn_res["viols"])), 4)])
    print(f"  CSV  saved -> {csv_path}")

    plot_training_curve(cb, ilp_ar, base_dir, sc_name)
    plot_comparison(ilp_ar, rand_res, dqn_res, base_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {base_dir}/training_curve.png")
    print(f"  {base_dir}/comparison.png")
    print(f"  {base_dir}/results.json")
    print(f"  {base_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
