from __future__ import annotations

"""Centralized default training-step configuration for all algorithms.

Edit this file when you want to change the default TOTAL_STEPS used by each algorithm.
Command-line overrides such as ``--total-timesteps`` still take precedence at runtime.
"""

GLOBAL_TOTAL_STEPS = 5_000_000

# Optional per-algorithm overrides. If a key is absent, GLOBAL_TOTAL_STEPS is used.
PROBLEM_TOTAL_STEPS: dict[str, int] = {
    "ppo": GLOBAL_TOTAL_STEPS,
    "ppo_mask": GLOBAL_TOTAL_STEPS,
    "ppo_lagrangian": GLOBAL_TOTAL_STEPS,
    "ppo_opt": GLOBAL_TOTAL_STEPS,
    "dqn": 3_000_000,
    "ddqn": 3_000_000,
}


def get_total_steps(problem_name: str) -> int:
    return int(PROBLEM_TOTAL_STEPS.get(problem_name, GLOBAL_TOTAL_STEPS))