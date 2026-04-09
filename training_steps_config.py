from __future__ import annotations

"""Centralized default training-step configuration for all trainable problems.

Edit this file when you want to change the default TOTAL_STEPS used by:
    - problem3_ppo
    - problem4_ppo_mask
    - problem5_ppo_lagrangian
    - problem6_ppo_opt
    - problem_dqn
    - problem_ddqn

Command-line overrides such as ``--total-timesteps`` still take precedence at runtime.
"""

GLOBAL_TOTAL_STEPS = 5_000_000

# Optional per-problem overrides. If a problem key is absent, GLOBAL_TOTAL_STEPS is used.
PROBLEM_TOTAL_STEPS: dict[str, int] = {
    "problem3_ppo": GLOBAL_TOTAL_STEPS,
    "problem4_ppo_mask": GLOBAL_TOTAL_STEPS,
    "problem5_ppo_lagrangian": GLOBAL_TOTAL_STEPS,
    "problem6_ppo_opt": GLOBAL_TOTAL_STEPS,
    "problem_dqn": 3_000_000,
    "problem_ddqn": 3_000_000,
}


def get_total_steps(problem_name: str) -> int:
    return int(PROBLEM_TOTAL_STEPS.get(problem_name, GLOBAL_TOTAL_STEPS))