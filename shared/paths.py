"""
paths.py — Single source of truth for where experiment artifacts live.

All algorithms, in every scenario, write to results/<scenario>/<algo>/ under
the project root. This keeps scenarios/ code-only; nothing under scenarios/
should ever create a results/ directory of its own.
"""

import hashlib
import json
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

PROGRESS_FILENAME = ".progress.json"

# Bump this on every tagged release (git tag vX.Y.Z) — embedded in saved
# model filenames so a model file is self-describing even if it's copied
# out of its results/<scenario>/<algo>/<run_id>/ directory.
VERSION = "0.1.1"


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


def new_run_id(algo_dir: Path) -> str:
    """8-bit run id (2 hex chars, 00-ff) unique among algo_dir's existing subdirs."""
    algo_dir.mkdir(parents=True, exist_ok=True)
    existing = {d.name for d in algo_dir.iterdir() if d.is_dir()}
    for _ in range(256):
        run_id = f"{secrets.randbelow(256):02x}"
        if run_id not in existing:
            return run_id
    raise RuntimeError(f"run id space exhausted under {algo_dir}")


def content_hash(key: str) -> str:
    """Deterministic 8-bit (2 hex char) hash of a cache key — same key always
    maps to the same filename, so content-addressed caches (like the shared
    ILP cache) don't need a fixed name and auto-bust when the key changes."""
    return hashlib.sha1(key.encode()).hexdigest()[:2]
