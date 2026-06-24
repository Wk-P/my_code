"""
compare_dqn_constraints.py — Quick (single-seed) comparison of constraint
handling strategies for DQN/DDQN on the LT scenario (N < M, ECU < SVC).

Background: the original LT results showed DQN/DDQN with 50-80% capacity /
conflict violation rates, far higher than EQ/GT. Investigation traced this to
two structural causes: (1) LT's combinatorics (N<M) make violations more
likely than EQ/GT's N=M setup, and (2) the LT env never terminates early, so
one episode can accumulate many violations, unlike EQ/GT's hard-terminate
design. This script retrains DQN/DDQN under three constraint_mode settings
("soft" = original, "hard" = terminate on violation, "lagrange" = dual-ascent
penalty weight) with a single seed and reduced step budget, to see whether
giving DQN/DDQN an actual constraint mechanism (instead of a fixed -2.0
penalty) closes the gap with MaskablePPO/Lagrange PPO/PPO+Repair.

Run:
    python compare_dqn_constraints.py --total-timesteps 400000
"""

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from problem2_ilp.objects import ECU, SVC
from problem_dqn.env import DQNEnv, LagrangeState as DQNLagrangeState
from problem_ddqn.env import DDQNEnv, LagrangeState as DDQNLagrangeState
from problem_ddqn.run_all import DDQN

import problem_dqn.config as DQNC
import problem_ddqn.config as DDQNC

MODES = ["soft", "hard", "lagrange"]

# Pseudo-modes layered on top of the base constraint_mode, with asymmetric
# penalty weights. Added because "hard" alone cut capacity violations to ~0
# but barely moved conflict violations (DQN/DDQN apparently learn to dodge
# capacity overflow much faster than conflicts under an equal -2.0/-2.0 split)
# — so here we make the conflict penalty heavier than capacity's to see if a
# stronger signal closes that gap.
MODE_CONFIG = {
    "soft":                dict(constraint_mode="soft",     cap_w=2.0, conf_w=2.0),
    "hard":                dict(constraint_mode="hard",     cap_w=2.0, conf_w=2.0),
    "lagrange":            dict(constraint_mode="lagrange", cap_w=2.0, conf_w=2.0),
    "hard_heavyconflict":  dict(constraint_mode="hard",     cap_w=2.0, conf_w=8.0),
}


def make_env_fn(env_cls, lagrange_cls, mode, seed, train_scenarios, prototype_caps, prototype_reqs):
    cfg = MODE_CONFIG[mode]

    def _make():
        import random
        random.seed(seed)
        lagrange_state = lagrange_cls() if cfg["constraint_mode"] == "lagrange" else None
        ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(prototype_caps)]
        svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(prototype_reqs)]
        env = env_cls(ecus, svcs, scenarios=train_scenarios,
                       constraint_mode=cfg["constraint_mode"], lagrange_state=lagrange_state,
                       cap_penalty_weight=cfg["cap_w"], conflict_penalty_weight=cfg["conf_w"])
        return Monitor(env)
    return _make


def train_and_eval(algo_name: str, mode: str, total_steps: int, device: str):
    is_ddqn = algo_name == "ddqn"
    C       = DDQNC if is_ddqn else DQNC
    env_cls = DDQNEnv if is_ddqn else DQNEnv
    lag_cls = DDQNLagrangeState if is_ddqn else DQNLagrangeState
    algo_cls = DDQN if is_ddqn else DQN

    torch.set_num_threads(8)
    n_envs = max(1, int(C.N_ENVS))
    proto_caps, proto_reqs, _ = C.SCENARIOS[C.SCENARIO_IDX]
    env = DummyVecEnv([
        make_env_fn(env_cls, lag_cls, mode, C.SEED + i, C.TRAIN_SCENARIOS, proto_caps, proto_reqs)
        for i in range(n_envs)
    ])

    model = algo_cls(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = C.DQN_LR if not is_ddqn else C.DDQN_LR,
        buffer_size            = C.DQN_BUFFER_SIZE if not is_ddqn else C.DDQN_BUFFER_SIZE,
        learning_starts        = C.DQN_LEARNING_STARTS if not is_ddqn else C.DDQN_LEARNING_STARTS,
        batch_size             = C.DQN_BATCH_SIZE if not is_ddqn else C.DDQN_BATCH_SIZE,
        tau                    = C.DQN_TAU if not is_ddqn else C.DDQN_TAU,
        gamma                  = C.DQN_GAMMA if not is_ddqn else C.DDQN_GAMMA,
        train_freq             = C.DQN_TRAIN_FREQ if not is_ddqn else C.DDQN_TRAIN_FREQ,
        gradient_steps          = C.DQN_GRADIENT_STEPS if not is_ddqn else C.DDQN_GRADIENT_STEPS,
        target_update_interval = C.DQN_TARGET_UPDATE if not is_ddqn else C.DDQN_TARGET_UPDATE,
        exploration_fraction   = C.DQN_EXPLORATION_FRACTION if not is_ddqn else C.DDQN_EXPLORATION_FRACTION,
        exploration_final_eps  = C.DQN_EXPLORATION_FINAL_EPS if not is_ddqn else C.DDQN_EXPLORATION_FINAL_EPS,
        policy_kwargs          = dict(net_arch=C.DQN_NET_ARCH if not is_ddqn else C.DDQN_NET_ARCH),
        device                 = device,
        verbose                = 0,
        seed                   = C.SEED,
    )
    t0 = time.time()
    model.learn(total_timesteps=total_steps)
    elapsed = time.time() - t0
    env.close()

    # Evaluation: deterministic policy on TEST_SCENARIOS, same constraint_mode
    # (mode only affects training-time reward shaping/termination; evaluation
    # always measures violations the same way: count + rate, episode runs to
    # the scenario's M steps unless the agent itself caused a hard-terminate).
    cfg = MODE_CONFIG[mode]
    ars, cap_viols, conflict_viols = [], [], []
    for scenario in C.TEST_SCENARIOS:
        caps, reqs, _ = scenario
        ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
        svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
        env_eval = env_cls(ecus, svcs, scenarios=[scenario], constraint_mode=cfg["constraint_mode"],
                            cap_penalty_weight=cfg["cap_w"], conflict_penalty_weight=cfg["conf_w"])
        obs, _ = env_eval.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env_eval.step(int(action))
        ars.append(info.get("ar", 0.0))
        cap_viols.append(int(info.get("capacity_violations", 0)))
        conflict_viols.append(int(info.get("conflict_violations", 0)))

    n_test = len(C.TEST_SCENARIOS)
    return {
        "algo": algo_name, "mode": mode, "elapsed_s": round(elapsed, 1),
        "n_test": n_test,
        "ar_mean": float(np.mean(ars)),
        "cap_viol_rate": float(np.mean(cap_viols)) ,
        "conflict_viol_rate": float(np.mean(conflict_viols)),
        "cap_viol_total": int(np.sum(cap_viols)),
        "conflict_viol_total": int(np.sum(conflict_viols)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=400_000)
    p.add_argument("--algos", nargs="+", default=["dqn", "ddqn"])
    p.add_argument("--modes", nargs="+", default=MODES)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  total_timesteps={args.total_timesteps:,}")

    results = []
    for algo in args.algos:
        for mode in args.modes:
            print(f"\n=== {algo.upper()} | constraint_mode={mode} ===")
            r = train_and_eval(algo, mode, args.total_timesteps, device)
            print(f"  AR={r['ar_mean']:.4f}  cap_viol/ep={r['cap_viol_rate']:.2f} "
                  f"conflict_viol/ep={r['conflict_viol_rate']:.2f}  "
                  f"({r['elapsed_s']:.1f}s, n_test={r['n_test']})")
            results.append(r)

    print(f"\n{'='*90}")
    print(f"  {'Algo':<6} {'Mode':<10} {'AR':<8} {'CapViol/ep':<12} {'ConflictViol/ep':<16} {'Time(s)'}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*12} {'-'*16} {'-'*7}")
    for r in results:
        print(f"  {r['algo']:<6} {r['mode']:<10} {r['ar_mean']:<8.4f} "
              f"{r['cap_viol_rate']:<12.2f} {r['conflict_viol_rate']:<16.2f} {r['elapsed_s']:.1f}")
    print(f"{'='*90}\n")

    import json
    out_path = HERE / "compare_dqn_constraints_results.json"
    existing = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    merged = {(r["algo"], r["mode"]): r for r in existing}
    merged.update({(r["algo"], r["mode"]): r for r in results})
    with open(out_path, "w") as f:
        json.dump(list(merged.values()), f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
