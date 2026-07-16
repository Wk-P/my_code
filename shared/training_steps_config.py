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
# lt's training curve (valid_placed/ep, added in v1.2.0) shows it hadn't
# plateaued yet at 2M steps -- still trending up when training cut off,
# unlike the Episode AR curve which plateaus early and misleadingly looks
# "done". eq/gt already hit 100% success_rate at 2M, so only lt needs the
# longer budget back.
SCENARIO_TOTAL_STEPS: dict[tuple[str, str], int] = {
    ("lt", "ppo_mask"): 5_000_000,
    ("lt", "ppo_lagrangian"): 5_000_000,
}


def get_total_steps(problem_name: str, scenario: str | None = None) -> int:
    if scenario is not None and (scenario, problem_name) in SCENARIO_TOTAL_STEPS:
        return int(SCENARIO_TOTAL_STEPS[(scenario, problem_name)])
    return int(PROBLEM_TOTAL_STEPS.get(problem_name, GLOBAL_TOTAL_STEPS))