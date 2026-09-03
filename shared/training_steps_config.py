from __future__ import annotations

"""Centralized default training-step configuration for all algorithms.

Edit this file when you want to change the default TOTAL_STEPS used by each algorithm.
Command-line overrides such as ``--total-timesteps`` still take precedence at runtime.
"""

GLOBAL_TOTAL_STEPS = 2_000_000

# Optional per-algorithm overrides. If a key is absent, GLOBAL_TOTAL_STEPS is used.
PROBLEM_TOTAL_STEPS: dict[str, int] = {
    "ppo": GLOBAL_TOTAL_STEPS,
    "ppo_mask": GLOBAL_TOTAL_STEPS,
    "ppo_lagrangian": GLOBAL_TOTAL_STEPS,
    "ppo_opt": GLOBAL_TOTAL_STEPS,
    "dqn": 2_000_000,
    "ddqn": 2_000_000,
}

# Per-(scenario, algorithm) overrides — takes precedence over PROBLEM_TOTAL_STEPS.
#
# IMPORTANT: steps must stay uniform ACROSS ALGORITHMS WITHIN one scenario --
# a controlled cross-algorithm comparison table is meaningless if some
# algorithms in it got more training budget than others (you can no longer
# tell whether a gap is the algorithm or the step count). lt is 5M for every
# algorithm listed below AND for ppo/ppo_opt/dqn/ddqn via explicit
# `--total-timesteps 5000000` in scripts/lt_other_algos_5seed.sh -- this dict
# only shows ppo_mask/ppo_lagrangian because those two are launched through
# run_all.py's own default (no CLI override), not because the other four run
# at a different budget.
#
# 2026-09-04: gt/dqn and gt/ddqn's eq_gt_migration_5seed batch showed they
# hadn't converged at 2M (Episode AR still trending up, conflict violation
# rate still trending down at the cutoff), unlike gt/ppo and
# gt/ppo_lagrangian which plateau by ~500k. Per the uniform-budget rule
# above, the fix is to move ALL of gt's algorithms to 5M together, not just
# the two that individually looked unconverged -- mirrors how lt did it.
# eq is left at 2M: all five of its algorithms are confirmed converged
# there, so there's no unconverged outlier forcing eq's budget up.
SCENARIO_TOTAL_STEPS: dict[tuple[str, str], int] = {
    ("lt", "ppo_mask"): 5_000_000,
    ("lt", "ppo_lagrangian"): 5_000_000,
    ("gt", "ppo_mask"): 5_000_000,
    ("gt", "ppo_lagrangian"): 5_000_000,
    ("gt", "ppo"): 5_000_000,
    ("gt", "ppo_opt"): 5_000_000,
    ("gt", "dqn"): 5_000_000,
    ("gt", "ddqn"): 5_000_000,
}


def get_total_steps(problem_name: str, scenario: str | None = None) -> int:
    if scenario is not None and (scenario, problem_name) in SCENARIO_TOTAL_STEPS:
        return int(SCENARIO_TOTAL_STEPS[(scenario, problem_name)])
    return int(PROBLEM_TOTAL_STEPS.get(problem_name, GLOBAL_TOTAL_STEPS))