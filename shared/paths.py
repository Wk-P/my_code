"""
paths.py — Single source of truth for where experiment artifacts live.

All algorithms, in every scenario, write to results/<scenario>/<algo>/ under
the project root. This keeps scenarios/ code-only; nothing under scenarios/
should ever create a results/ directory of its own.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def results_dir(scenario: str, *parts: str) -> Path:
    """results/<scenario>/<*parts>, e.g. results_dir("lt", "ppo_opt")."""
    return PROJECT_ROOT.joinpath("results", scenario, *parts)
