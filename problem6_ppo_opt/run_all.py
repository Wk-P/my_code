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
import sys, time, json
import numpy as np
import matplotlib

import timer_utils
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch
import yaml
import pulp
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import config as C
from problem6_ppo_opt.env import P3Env


def resolve_device(cfg: str) -> str:
    if cfg != "auto":
        return cfg
    if torch.cuda.is_available():
        print(f"[CUDA] {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[CPU] No CUDA GPU, using CPU")
    return "cpu"
from problem2_ilp.objects import ECU, SVC


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Load scenario from YAML
# ══════════════════════════════════════════════════════════════════════════════

def load_scenario():
    with open(C.YAML_CONFIG, "r") as f:
        data = yaml.safe_load(f)
    scenario = data["scenarios"][C.SCENARIO_IDX]

    ecus     = [ECU(s["name"], s["capacity"])   for s in scenario["ECUs"]]
    services = [SVC(s["name"], s["requirement"]) for s in scenario["SVCs"]]

    N = len(ecus)
    M = len(services)
    print(f"Loaded: {scenario['name']}  |  N={N} ECUs  M={M} SVCs")
    print(f"  ECU capacities : {[e.capacity for e in ecus]}")
    print(f"  SVC requirements: {[s.requirement for s in services]}")
    return ecus, services, scenario["name"]


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — ILP optimal solution
# ══════════════════════════════════════════════════════════════════════════════

def solve_ilp(ecus: list[ECU], services: list[SVC]) -> dict:
    """Solve via PuLP; return avg_utilization and the allocation map."""
    N = len(ecus)
    M = len(services)
    e_list = [e.capacity   for e in ecus]
    n_list = [s.requirement for s in services]

    prob = pulp.LpProblem("P3_ILP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(M), range(N)), cat="Binary")

    # Objective: maximise total utilisation (consistent with problem2_ilp ILP)
    prob += pulp.lpSum(x[i][j] * n_list[i] / e_list[j]
                       for i in range(M) for j in range(N))

    for i in range(M):
        prob += pulp.lpSum(x[i][j] for j in range(N)) == 1          # every service assigned to exactly one ECU
    for j in range(N):
        prob += pulp.lpSum(x[i][j] for i in range(M)) <= 1          # each ECU hosts at most one service
    for i in range(M):
        for j in range(N):
            if n_list[i] > e_list[j]:
                prob += x[i][j] == 0                                 # infeasible pair: demand exceeds capacity
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
        "status":          pulp.LpStatus[prob.status],
        "avg_utilization": avg_util,
        "total_utilization": total_util,
        "active_ecus":     active_ecus,
        "allocation":      alloc,
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

    # ─ 1. Shared cache written by problem2_ilp/optimal_solution/main.py ─
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
#  Step 3 & 5 — Evaluation (Random / PPO)
# ══════════════════════════════════════════════════════════════════════════════

def run_episodes(ecus, services, policy_fn, n_eps: int):
    """
    Run n_eps episodes on a fixed problem instance.
    Episodes always complete (M steps, no early termination in P3).
    policy_fn(obs) -> int
    """
    env = P3Env(ecus, services, scenarios=C.SCENARIOS)
    ars, cap_viols, dup_viols = [], [], []

    for _ in range(n_eps):
        obs, _ = env.reset()
        done   = False
        info   = {}
        while not done:
            obs, _, done, _, info = env.step(policy_fn(obs))
        ars.append(info["ar"])
        cap_viols.append(info["capacity_violations"])
        dup_viols.append(info["single_service_violations"])

    return {
        "ars":        np.array(ars),
        "cap_viols":  np.array(cap_viols),
        "dup_viols":  np.array(dup_viols),
        "tot_viols":  np.array(cap_viols) + np.array(dup_viols),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — PPO training
# ══════════════════════════════════════════════════════════════════════════════

class P3Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_ars        : list[float] = []
        self.episode_violations : list[int]   = []
        self.timesteps_at_ep    : list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_ars.append(float(info["episode"]["r"]))
                self.episode_violations.append(int(info.get("total_violations", 0)))
                self.timesteps_at_ep.append(self.num_timesteps)
        return True


def train_ppo(ecus, services) -> tuple[PPO, P3Callback]:
    env   = Monitor(P3Env(ecus, services, scenarios=C.SCENARIOS))
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
    print(f"  Training done  {elapsed:.1f}s | {n_ep} episodes | AR(last50)={last50:.4f}")
    return model, cb


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def moving_avg(arr, w):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr, 0
    return np.convolve(arr, np.ones(w) / w, mode="valid"), w - 1


def plot_training_curve(cb: P3Callback, ilp_ar: float, outdir: Path, scenario_name: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ts = np.array(cb.timesteps_at_ep)
    N  = len(cb.episode_ars)

    # ── AR ────────────────────────────────────────────────────────────────────
    sm, off = moving_avg(cb.episode_ars, C.SMOOTH_W)
    ax1.plot(ts, cb.episode_ars, color="steelblue", alpha=0.2, linewidth=0.8)
    ax1.plot(ts[off:off+len(sm)], sm, color="steelblue", linewidth=2,
             label=f"PPO (smoothed w={C.SMOOTH_W})")
    ax1.axhline(ilp_ar, color="red", linestyle="--", linewidth=1.5,
                label=f"ILP Optimal  AR={ilp_ar:.4f}")
    ax1.set_ylabel("Episode AR", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"PPO Training Curve — {scenario_name}  ({C.TOTAL_STEPS:,} steps)", fontsize=12)
    ax1.grid(alpha=0.3)

    # ── Violations ────────────────────────────────────────────────────────────
    sm_v, off_v = moving_avg(cb.episode_violations, C.SMOOTH_W)
    ax2.plot(ts, cb.episode_violations, color="tomato", alpha=0.2, linewidth=0.8)
    ax2.plot(ts[off_v:off_v+len(sm_v)], sm_v, color="tomato", linewidth=2,
             label=f"violations/ep (smoothed)")
    ax2.set_ylabel("Constraint Violations / Episode", fontsize=11)
    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = outdir / "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_comparison(ilp_ar, rand_res, ppo_res, outdir: Path, scenario_name: str):
    M_steps = len(rand_res["ars"])   # not needed but keep for label

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"P2(ILP) vs Random vs P3(PPO) — {scenario_name}", fontsize=13, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels = ["ILP\n(Optimal)", "Random\nBaseline", "PPO\n(P3, no constraint)"]

    # ── Left: AR box plot ─────────────────────────────────────────────────────
    ax = axes[0]
    # ILP is a single deterministic value — shown as a dashed horizontal line
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

    # ILP: dashed red line + diamond marker
    ax.axhline(ilp_ar, color=colors[0], linestyle="--", linewidth=2, alpha=0.9,
               label=f"ILP  AR={ilp_ar:.4f}")
    ax.plot(1, ilp_ar, marker="D", color=colors[0], markersize=10, zorder=5)

    # annotate mean values
    for pos, data, color in zip([2, 3], [rand_ars, ppo_ars], colors[1:]):
        mv = np.mean(data)
        ax.text(pos, mv + 0.02, f"μ={mv:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Resource Utilisation (AR)", fontsize=11)
    ax.set_title("AR Distribution", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # ── Right: Violation rate bar ─────────────────────────────────────────────
    ax2 = axes[1]
    N_svc = ppo_res["ars"].shape[0]   # just number of eval episodes
    # ILP violation rate = 0 (guaranteed by hard constraints)
    vr_means = [
        0.0,
        np.mean(rand_res["tot_viols"]),
        np.mean(ppo_res["tot_viols"]),
    ]
    vr_stds = [
        0.0,
        np.std(rand_res["tot_viols"]),
        np.std(ppo_res["tot_viols"]),
    ]
    bars = ax2.bar(labels, vr_means, color=colors, alpha=0.75,
                   yerr=vr_stds, capsize=5, ecolor="black")
    for bar, v in zip(bars, vr_means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 v + max(vr_means) * 0.02 + 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Avg Constraint Violations per Episode", fontsize=11)
    ax2.set_title("Constraint Violations", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

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
    print(f"\n{'='*60}")
    print(f"  P3 run_all.py  —  RL WITHOUT constraint enforcement")
    print(f"  Config : {C.YAML_CONFIG.name}  Scenario idx={C.SCENARIO_IDX}")
    print(f"{'='*60}\n")

    # ── 1. Load scenario ─────────────────────────────────────────────────────
    ecus, services, sc_name = load_scenario()
    N = len(ecus)
    M = len(services)

    # ── 2. ILP (all scenarios) ────────────────────────────────────────────
    print(f"\n[1/4] Solving ILP for all {len(C.SCENARIOS)} scenarios ...")
    ilp_ar, ilp_per_sc = solve_ilp_all_scenarios()
    print(f"  ILP mean AR across {len(C.SCENARIOS)} scenarios: {ilp_ar:.4f}")

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
    print(f"  Violations/ep mean={np.mean(rand_res['tot_viols']):.2f}")

    # ── 4. PPO training ──────────────────────────────────────────────────────
    print(f"\n[3/4] PPO training ({C.TOTAL_STEPS:,} steps) ...")
    model, cb = train_ppo(ecus, services)
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
    print(f"  Violations/ep mean={np.mean(ppo_res['tot_viols']):.2f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    M_total = M
    print(f"\n{'='*62}")
    print(f"  {'Method':<24} {'AR (mean±std)':<22} {'Viol/ep':<10}")
    print(f"  {'-'*24} {'-'*22} {'-'*10}")
    print(f"  {'ILP (Optimal)':<24} {ilp_ar:.4f} ± 0.0000       {'0':<10}")
    r_v = np.mean(rand_res['tot_viols'])
    p_v = np.mean(ppo_res['tot_viols'])
    print(f"  {'Random Baseline':<24} "
          f"{np.mean(rand_res['ars']):.4f} ± {np.std(rand_res['ars']):.4f}   "
          f"  {r_v:<10.2f}")
    print(f"  {'PPO (P3, no constr)':<24} "
          f"{np.mean(ppo_res['ars']):.4f} ± {np.std(ppo_res['ars']):.4f}   "
          f"  {p_v:<10.2f}")
    print(f"{'='*62}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    log = {
        "scenario": sc_name,
        "N": N, "M": M,
        "ilp":    {
            "ar": round(ilp_ar, 6),
            "ar_per_scenario": [round(r["avg_utilization"], 6) for r in ilp_per_sc],
            "violations": 0,
        },
        "random": {
            "ar_mean":   round(float(np.mean(rand_res["ars"])), 6),
            "ar_std":    round(float(np.std(rand_res["ars"])),  6),
            "viol_mean": round(float(r_v), 4),
        },
        "ppo": {
            "ar_mean":   round(float(np.mean(ppo_res["ars"])), 6),
            "ar_std":    round(float(np.std(ppo_res["ars"])),  6),
            "viol_mean": round(float(p_v), 4),
        },
        "training": {
            "total_steps":  C.TOTAL_STEPS,
            "n_episodes":   len(cb.episode_ars),
            "ar_last50":    round(float(np.mean(cb.episode_ars[-50:])), 6),
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
        writer.writerow(["method", "ar_mean", "ar_std", "viol_per_ep"])
        writer.writerow(["ILP (Optimal)",       round(ilp_ar, 6), 0.0,                          0.0])
        writer.writerow(["Random Baseline",      round(float(np.mean(rand_res["ars"])), 6),
                         round(float(np.std(rand_res["ars"])),  6), round(float(r_v), 4)])
        writer.writerow(["PPO (P3, no constr)",  round(float(np.mean(ppo_res["ars"])),  6),
                         round(float(np.std(ppo_res["ars"])),   6), round(float(p_v), 4)])
    print(f"  CSV  saved → {csv_path}")
    plot_comparison(ilp_ar, rand_res, ppo_res, run_dir, sc_name)

    print("\nAll done! Output files:")
    print(f"  {run_dir}/training_curve.png")
    print(f"  {run_dir}/comparison.png")
    print(f"  {run_dir}/results.json")
    print(f"  {run_dir}/summary.csv\n")


if __name__ == "__main__":
    main()
