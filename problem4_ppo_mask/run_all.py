"""
run_all.py — One-shot P4 full pipeline:

  1. Load scenario from YAML (same as P2/P3)
  2. Solve with ILP (PuLP)                        -> ilp_ar (optimal upper bound)
  3. Evaluate random policy (with action masking)  -> random_ars
  4. Train MaskablePPO (with action masking)       -> training curve
  5. Evaluate trained MaskablePPO                  -> ppo_ars
  6. Produce plots:
       - comparison.png     — 3-way AR box plot + services placed
       - training_curve.png — AR during training

P4 Design: constraints enforced via action masking -> 0 violations guaranteed.

Run:
    python problem4_single/run_all.py
"""

import datetime
import csv
import sys, time, json, functools
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
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import config as C
from env_p4 import P4Env
from problem2_ilp.objects import ECU, SVC


def _make_p4_env(seed: int) -> Monitor:
    """Module-level factory (picklable for SubprocVecEnv on Windows)."""
    import random
    random.seed(seed)
    caps, reqs = C.SCENARIOS[C.SCENARIO_IDX]
    ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
    services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
    env = P4Env(ecus, services, scenarios=C.SCENARIOS)
    return Monitor(env)


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

    prob = pulp.LpProblem("P4_ILP", pulp.LpMaximize)
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
                "services": svcs,
                "utilization": sum(n_list[i] for i in svcs) / e_list[j],
                "capacity": e_list[j],
                "demand": sum(n_list[i] for i in svcs),
            }
    total_util  = pulp.value(prob.objective) or 0.0
    active_ecus = len(alloc)
    avg_util    = total_util / active_ecus if active_ecus > 0 else 0.0
    return {
        "status": pulp.LpStatus[prob.status],
        "avg_utilization": avg_util,
        "total_utilization": total_util,
        "active_ecus": active_ecus,
        "allocation": alloc,
    }


def solve_ilp_all_scenarios():
    """Return (mean_ar, per_scenario_results) for all scenarios.
    Priority: 1) shared p2 cache  2) own local cache  3) compute from scratch.
    """
    cache_key = f"{C.YAML_CONFIG.name}__n{len(C.SCENARIOS)}"

    def _load_cache(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cache(path, data):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.replace(path)

    # ─ 1. Shared cache written by problem2_single/optimal_solution/main.py ─
    shared_cache = C.YAML_CONFIG.parent.parent / "results" / "ilp_cache.json"
    if shared_cache.exists():
        cache = _load_cache(shared_cache)
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
        cache = _load_cache(cache_path)
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
        _save_cache(cache_path, {"key": cache_key, "results": results})

    print(f"    [cache] Saved to {cache_path}")
    ars = [r["avg_utilization"] for r in results]
    return float(np.mean(ars)), results


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 & 5 — Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps):
    """policy_fn(obs, mask) -> int"""
    env = P4Env(ecus, services, scenarios=C.SCENARIOS)
    ars, placed_list = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            mask = env.action_masks()
            if not np.any(mask):
                break
            obs, _, done, _, info = env.step(policy_fn(obs, mask))
        ars.append(info.get("ar", 0.0))
        placed_list.append(info.get("services_placed", 0))
    return {
        "ars":    np.array(ars),
        "placed": np.array(placed_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Training
# ══════════════════════════════════════════════════════════════════════════════

class P4Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_ars:     list[float] = []
        self.episode_placed:  list[int]   = []
        self.timesteps_at_ep: list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_ars.append(float(info["episode"]["r"]))
                self.episode_placed.append(int(info.get("services_placed", 0)))
                self.timesteps_at_ep.append(self.num_timesteps)
        return True


def train_maskppo(ecus, services):
    n_envs = 6
    env = SubprocVecEnv(
        [functools.partial(_make_p4_env, C.SEED + i) for i in range(n_envs)],
        start_method="spawn",
    )

    cb  = P4Callback()
    model = MaskablePPO(
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
        device        = resolve_device(C.DEVICE),
        verbose       = 0,
        seed          = C.SEED,
    )
    t0 = time.time()
    model.learn(total_timesteps=C.TOTAL_STEPS, callback=cb)
    elapsed = time.time() - t0
    env.close()

    n_ep   = len(cb.episode_ars)
    last50 = np.mean(cb.episode_ars[-50:]) if n_ep >= 50 else np.mean(cb.episode_ars)
    last50_p = np.mean(cb.episode_placed[-50:]) if n_ep >= 50 else np.mean(cb.episode_placed)
    print(f"  Training done  {elapsed:.1f}s | {n_ep} eps "
          f"| AR(last50)={last50:.4f} | placed(last50)={last50_p:.1f}/{C.M}")
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

    sm, off = moving_avg(cb.episode_ars, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_ars, color="seagreen", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off:off+len(sm)], sm, color="seagreen", linewidth=2,
             label=f"MaskablePPO (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"P4 MaskablePPO Training — {scenario_name}  ({C.TOTAL_STEPS:,} steps)",
                  fontsize=12)
    ax1.grid(alpha=0.3)

    sm_p, off_p = moving_avg(cb.episode_placed, C.SMOOTH_W)
    ax2.plot(ts, cb.episode_placed, color="royalblue", alpha=0.2, linewidth=0.8)
    ax2.plot(ts[off_p:off_p+len(sm_p)], sm_p, color="royalblue", linewidth=2,
             label="services placed/ep")
    ax2.axhline(C.M, color="red", linestyle="--", alpha=0.5, label=f"M={C.M}")
    ax2.set_ylabel("Services Placed", fontsize=11)
    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.set_ylim(0, C.M + 1)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_comparison(ilp_ar, rand_res, ppo_res, outdir, scenario_name):
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels = ["ILP\n(Optimal)", "Random\n(masked)", "MaskablePPO\n(P4)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"P2(ILP) vs Random(masked) vs P4(MaskablePPO) — {scenario_name}",
                 fontsize=13, fontweight="bold")

    # AR
    ax = axes[0]
    bp = ax.boxplot(
        [rand_res["ars"], ppo_res["ars"]],
        positions=[2, 3], widths=0.5, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors[1:]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.axhline(ilp_ar, color=colors[0], linestyle="--", linewidth=2, alpha=0.9,
               label=f"ILP  AR={ilp_ar:.4f}")
    ax.plot(1, ilp_ar, marker="D", color=colors[0], markersize=10, zorder=5)

    for pos, data, color in zip([2, 3], [rand_res["ars"], ppo_res["ars"]], colors[1:]):
        mv = np.mean(data)
        ax.text(pos, mv + 0.02, f"mu={mv:.3f}", ha="center", fontsize=9,
                fontweight="bold", color=color)

    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax.set_title("AR Distribution (0 violations)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Services placed
    ax2 = axes[1]
    pl_means = [C.M, np.mean(rand_res["placed"]), np.mean(ppo_res["placed"])]
    pl_stds  = [0.0, np.std(rand_res["placed"]), np.std(ppo_res["placed"])]
    bars = ax2.bar(labels, pl_means, color=colors, alpha=0.75,
                   yerr=pl_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, pl_means):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                 f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax2.axhline(C.M, color="gray", linestyle=":", alpha=0.4)
    ax2.set_ylim(0, C.M + 2)
    ax2.set_ylabel("Services Placed per Episode", fontsize=11)
    ax2.set_title("Placement Completeness", fontsize=11)
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
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  P4 run_all.py  —  RL WITH action masking (0 violations)")
    print(f"  Config : {C.YAML_CONFIG.name}  Scenario idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")

    # 1. Load scenario
    ecus, services, sc_name = load_scenario()
    N, M = len(ecus), len(services)

    # 2. ILP (all scenarios)
    print(f"\n[1/4] Solving ILP for all {len(C.SCENARIOS)} scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios()
    print(f"  ILP mean AR across {len(C.SCENARIOS)} scenarios: {ilp_ar:.4f}")

    # 3. Random (masked)
    print(f"\n[2/4] Random masked baseline ({C.EVAL_EPS} episodes) ...")
    np.random.seed(C.SEED)
    rand_res = run_episodes(
        ecus, services,
        policy_fn=lambda obs, mask: int(np.random.choice(np.where(mask)[0]))
                                    if np.any(mask) else 0,
        n_eps=C.EVAL_EPS,
    )
    print(f"  Random AR  mean={np.mean(rand_res['ars']):.4f}  "
          f"std={np.std(rand_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(rand_res['placed']):.1f}/{M}")

    # 4. MaskablePPO training
    print(f"\n[3/4] MaskablePPO training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_maskppo(ecus, services)
    model.save(str(C.MODEL_PATH))
    print(f"  Model saved -> {C.MODEL_PATH}.zip")

    # 5. MaskablePPO evaluation
    print(f"\n[4/4] MaskablePPO evaluation ({C.EVAL_EPS} episodes, deterministic) ...")
    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return int(action)
    ppo_res = run_episodes(ecus, services, ppo_policy, C.EVAL_EPS)
    print(f"  PPO AR  mean={np.mean(ppo_res['ars']):.4f}  "
          f"std={np.std(ppo_res['ars']):.4f}")
    print(f"  Placed/ep  mean={np.mean(ppo_res['placed']):.1f}/{M}")

    # Summary
    print(f"\n{'='*66}")
    print(f"  {'Method':<28} {'AR (mean+/-std)':<24} {'Placed':<10} {'Viol'}")
    print(f"  {'-'*28} {'-'*24} {'-'*10} {'-'*5}")
    print(f"  {'ILP (Optimal)':<28} {ilp_ar:.4f} +/- 0.0000       "
          f"{M}/{M:<7} 0")
    print(f"  {'Random (masked)':<28} "
          f"{np.mean(rand_res['ars']):.4f} +/- {np.std(rand_res['ars']):.4f}   "
          f"  {np.mean(rand_res['placed']):.1f}/{M:<4}  0")
    print(f"  {'MaskablePPO (P4)':<28} "
          f"{np.mean(ppo_res['ars']):.4f} +/- {np.std(ppo_res['ars']):.4f}   "
          f"  {np.mean(ppo_res['placed']):.1f}/{M:<4}  0")
    print(f"{'='*66}\n")

    # Save JSON
    log = {
        "scenario": sc_name, "N": N, "M": M,
        "ilp": {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "random_masked": {
            "ar_mean":   round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":    round(float(np.std(rand_res["ars"])), 6),
            "placed_mean": round(float(np.mean(rand_res["placed"])), 2),
            "violations": 0,
        },
        "maskable_ppo": {
            "ar_mean":   round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":    round(float(np.std(ppo_res["ars"])), 6),
            "placed_mean": round(float(np.mean(ppo_res["placed"])), 2),
            "violations": 0,
        },
        "training": {
            "total_steps": C.TOTAL_STEPS,
            "n_episodes":  len(cb.episode_ars),
            "ar_last50":   round(float(np.mean(cb.episode_ars[-50:])), 6),
        },
    }

    base_path = C.OUTDIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_path.mkdir(parents=True, exist_ok=True)
    log_path = base_path / "results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  JSON saved -> {log_path}")

    # Save CSV summary
    csv_path = base_path / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ar_mean", "ar_std", "placed_mean", "violations"])
        writer.writerow(["ILP (Optimal)",     round(ilp_ar, 6), 0.0, M, 0])
        writer.writerow(["Random (masked)",   round(float(np.mean(rand_res["ars"])), 6),
                         round(float(np.std(rand_res["ars"])), 6),
                         round(float(np.mean(rand_res["placed"])), 2), 0])
        writer.writerow(["MaskablePPO (P4)",  round(float(np.mean(ppo_res["ars"])),  6),
                         round(float(np.std(ppo_res["ars"])),  6),
                         round(float(np.mean(ppo_res["placed"])), 2), 0])
    print(f"  CSV  saved -> {csv_path}")

    # Plots
    plot_training_curve(cb, ilp_ar, base_path, sc_name)
    plot_comparison(ilp_ar, rand_res, ppo_res, base_path, sc_name)

    print("\nAll done! Output files:")
    print(f"  {base_path}/training_curve.png")
    print(f"  {base_path}/comparison.png")
    print(f"  {base_path}/results.json")
    print(f"  {base_path}/summary.csv\n")


if __name__ == "__main__":
    main()
