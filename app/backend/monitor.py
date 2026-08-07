"""
monitor.py — standalone watchdog process for "is a background training run
still making progress, or has it stalled?"

main.py's /api/progress is stateless: every request re-derives "running"
straight from `ps`, so it can tell you a process exists but has no memory of
whether its step count actually moved since last time. Detecting "stuck"
(process alive, but not progressing) needs exactly that memory across polls,
so it lives in a separate long-running loop instead of inside the
request/response cycle.

Writes app/backend/monitor_state.json every POLL_INTERVAL_SECONDS, read
(never written) by main.py's /api/progress. Read-only over training
state — like main.py, this never imports or executes anything under
scenarios/ or shared/; it only reads `ps` and progress files main.py already
knows how to read (reused directly from main.py to avoid duplicating the
process/log-matching logic).

Run standalone:
    .venv/bin/python -m app.backend.monitor
Intended to run under systemd (Restart=always) so it survives crashes and
comes back after reboot without anyone remembering to restart it by hand —
see the my-code-monitor.service unit installed alongside my-code-panel.service.
"""

import json
import time
from pathlib import Path

from app.backend.main import (
    SCENARIOS,
    _ps_snapshot,
    _match_scenario_algo,
    _read_live_progress,
)

APP_DIR = Path(__file__).parent
STATE_FILE = APP_DIR / "monitor_state.json"

POLL_INTERVAL_SECONDS = 30
# No step-count movement for this long while a process is alive -> "stuck".
STUCK_SECONDS = 15 * 60
# No .progress.json has appeared at all yet for this long -> also "stuck"
# (covers e.g. a hang before the first training callback tick).
NO_PROGRESS_FILE_STUCK_SECONDS = 15 * 60


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(STATE_FILE)


def _now() -> float:
    return time.time()


def tick(prev_state: dict) -> dict:
    """One poll: compare current `ps`/progress snapshot against prev_state
    (as loaded from STATE_FILE) and return the new state to persist."""
    procs = _ps_snapshot()
    new_state = {}

    for scenario in SCENARIOS:
        proc = next((p for p in procs if _match_scenario_algo(p["cmd"], p["pid"])[0] == scenario), None)
        prev = prev_state.get(scenario, {})

        if proc is None:
            # Nothing running here right now — no "stuck" concept applies;
            # main.py's own "running": false path already covers this case.
            new_state[scenario] = {"status": "idle", "checked_at": _now()}
            continue

        _, algo = _match_scenario_algo(proc["cmd"], proc["pid"])
        progress = _read_live_progress(scenario, algo)
        step = progress["step"] if progress else None

        same_pid = prev.get("pid") == proc["pid"]
        same_step = same_pid and step is not None and prev.get("last_step") == step

        if same_step:
            last_progress_at = prev.get("last_progress_at", _now())
        else:
            last_progress_at = _now()

        stalled_seconds = _now() - last_progress_at
        if step is None:
            # Never seen a progress tick for this pid yet.
            first_seen_at = prev.get("first_seen_at", _now()) if same_pid else _now()
            no_progress_seconds = _now() - first_seen_at
            status = "stuck" if no_progress_seconds > NO_PROGRESS_FILE_STUCK_SECONDS else "starting"
            new_state[scenario] = {
                "status": status,
                "pid": proc["pid"],
                "algo": algo,
                "first_seen_at": first_seen_at,
                "checked_at": _now(),
            }
            continue

        status = "stuck" if stalled_seconds > STUCK_SECONDS else "running"
        new_state[scenario] = {
            "status": status,
            "pid": proc["pid"],
            "algo": algo,
            "last_step": step,
            "last_progress_at": last_progress_at,
            "stalled_seconds": round(stalled_seconds),
            "checked_at": _now(),
        }

    return new_state


def run_forever() -> None:
    print(f"[monitor] starting, polling every {POLL_INTERVAL_SECONDS}s, "
          f"stuck threshold {STUCK_SECONDS}s, state file {STATE_FILE}")
    while True:
        try:
            prev_state = _load_state()
            new_state = tick(prev_state)
            _save_state(new_state)
        except Exception as e:  # watchdog must never die from a transient read error
            print(f"[monitor] tick failed: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
