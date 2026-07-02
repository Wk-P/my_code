"""
paths.py — Single source of truth for where experiment artifacts live.

All algorithms, in every scenario, write to results/<scenario>/<algo>/ under
the project root. This keeps scenarios/ code-only; nothing under scenarios/
should ever create a results/ directory of its own.
"""

import hashlib
import json
import os
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

PROGRESS_FILENAME = ".progress.json"
EXP_ID_ENV_VAR = "EXP_ID"

# Bump this on every tagged release (git tag vX.Y.Z) — embedded in saved
# model filenames so a model file is self-describing even if it's copied
# out of its results/<scenario>/<algo>/<exp_id>/ directory.
VERSION = "0.2.0"


def results_dir(scenario: str, *parts: str) -> Path:
    """results/<scenario>/<*parts>, e.g. results_dir("lt", "ppo_opt")."""
    return PROJECT_ROOT.joinpath("results", scenario, *parts)


def write_progress(algo_dir: Path, **fields) -> None:
    """Overwrite results/<scenario>/<algo>/.progress.json with the latest
    training progress. Written directly by the training callback instead of
    being scraped from stdout — stdout is block-buffered whenever it's
    redirected to a file (not a TTY), so log-tailing for live progress is
    unreliable; a small file write on every callback tick is not."""
    algo_dir.mkdir(parents=True, exist_ok=True)
    path = algo_dir / PROGRESS_FILENAME
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(fields, f)
    tmp.replace(path)


def _used_exp_ids() -> set[str]:
    """All exp_ids already in use anywhere under results/ — a run directory
    one level below any results/<scenario>/<algo>/ dir."""
    used = set()
    if not RESULTS_ROOT.is_dir():
        return used
    for scenario_dir in RESULTS_ROOT.iterdir():
        if not scenario_dir.is_dir():
            continue
        for algo_dir in scenario_dir.iterdir():
            if not algo_dir.is_dir():
                continue
            used.update(d.name for d in algo_dir.iterdir() if d.is_dir())
    return used


def new_exp_id() -> str:
    """Full 8-digit hex exp_id (32-bit, 4 random bytes), unique across all
    of results/.

    One exp_id identifies one whole experiment *batch* — e.g. eq+gt+lt all
    launched together share the same id — not one per algo. The orchestrating
    launch script (scripts/start_experiment.sh) calls this once and exports
    it as $EXP_ID; every scenario/algo process in that batch picks it up via
    resolve_exp_id() instead of minting its own.
    """
    used = _used_exp_ids()
    for _ in range(4096):
        exp_id = secrets.token_hex(4)
        if exp_id not in used:
            return exp_id
    raise RuntimeError("could not find a free exp id under results/ after 4096 tries")


def resolve_exp_id(algo_dir: Path) -> str:
    """The exp_id for the run about to be written under algo_dir.

    Prefers $EXP_ID (set by the launch script so a whole batch of
    scenarios/algos shares one id); falls back to minting a fresh one for
    ad-hoc standalone runs (e.g. running a single run_all.py by hand)."""
    algo_dir.mkdir(parents=True, exist_ok=True)
    env_id = os.environ.get(EXP_ID_ENV_VAR)
    if env_id:
        return env_id
    return new_exp_id()


def content_hash(key: str) -> str:
    """Deterministic 8-bit (2 hex char) hash of a cache key — same key always
    maps to the same filename, so content-addressed caches (like the shared
    ILP cache) don't need a fixed name and auto-bust when the key changes."""
    return hashlib.sha1(key.encode()).hexdigest()[:2]
