"""
my-code experiment dashboard — read-only view over results/<scenario>/<algo>/<run>/
plus best-effort process/log introspection for live training progress.

This module never imports or executes anything under scenarios/ or shared/ —
it only reads files (results.json, PNGs, log files) and OS process state
(ps, /proc). It cannot affect running or future training runs.

Serves:
  GET  /api/results                                  summary table (latest run per scenario/algo)
  GET  /api/results/{scenario}/{algo}/{run}/{file}    training_curve.png / comparison.png
  GET  /api/history/{scenario}/{algo}                 all historical runs for one algo
  GET  /api/progress                                  live training progress per scenario
  GET  /api/system                                    CPU/load info, grouped by pinned core range
  GET  /                                              single-page dashboard (vanilla JS, no build step)
"""

import json
import re
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

APP_DIR      = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
SCENARIOS    = ["eq", "gt", "lt"]

app = FastAPI(title="my-code experiment dashboard")
app.mount("/assets", StaticFiles(directory=APP_DIR / "static" / "assets"), name="assets")


def _algo_key(data: dict) -> str | None:
    reserved = ("scenario", "prototype_scenario", "scenario_count",
                "train_count", "test_count", "N", "M", "ilp", "training",
                "feasibility", "created_at")
    return next((k for k in data.keys() if k not in reserved), None)


def _viol_rate(algo_eval: dict, test_count) -> float | None:
    """Violation-rate field naming isn't consistent across algorithms
    (ppo: conflict_viol_rate_mean, ppo_mask: violations count that's
    always 0 by design, dqn/ddqn: viol_rate) — try each in turn."""
    for key in ("viol_rate", "viol_rate_mean", "conflict_viol_rate_mean"):
        if key in algo_eval:
            return algo_eval[key]
    if "violations" in algo_eval and test_count:
        return algo_eval["violations"] / test_count
    return None


def _row_from_run(scenario: str, algo: str, run_dir: Path) -> dict | None:
    try:
        data = json.loads((run_dir / "results.json").read_text())
    except (json.JSONDecodeError, OSError):
        return None
    algo_key  = _algo_key(data)
    algo_eval = data.get(algo_key, {}) if algo_key else {}
    training  = data.get("training", {})
    ilp       = data.get("ilp", {})
    return {
        "scenario":      scenario,
        "algo":          algo,
        "run":           run_dir.name,
        "N":             data.get("N"),
        "M":             data.get("M"),
        "train_count":   data.get("train_count"),
        "test_count":    data.get("test_count"),
        "ilp_ar":        ilp.get("ar"),
        "test_ar_mean":  algo_eval.get("ar_mean"),
        "test_ar_std":   algo_eval.get("ar_std"),
        "test_viol_rate":      _viol_rate(algo_eval, data.get("test_count")),
        "test_cap_viol_total":      algo_eval.get("cap_viol_total"),
        "test_conflict_viol_total": algo_eval.get("conflict_viol_total"),
        "train_ar_last50": training.get("ar_last50"),
        "train_steps":     training.get("total_steps"),
        "train_episodes":  training.get("n_episodes"),
    }


_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")


def _sort_key(run_dir: Path) -> str:
    """Run dirs used to be named as timestamps (lexicographic == chronological);
    now they're random 8-bit hash ids, so chronology has to come from
    results.json's created_at field, with a fallback that normalizes old
    timestamp dir names into the same sortable ISO-ish shape."""
    try:
        data = json.loads((run_dir / "results.json").read_text())
    except (json.JSONDecodeError, OSError):
        data = {}
    if data.get("created_at"):
        return data["created_at"]
    m = _TS_RE.match(run_dir.name)
    if m:
        y, mo, d, h, mi, s = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
    return run_dir.name


def _run_dirs(algo_dir: Path) -> list[Path]:
    runs = [d for d in algo_dir.iterdir() if d.is_dir() and (d / "results.json").exists()]
    return sorted(runs, key=_sort_key)


def _collect_results() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        scenario_dir = RESULTS_ROOT / scenario
        if not scenario_dir.is_dir():
            continue
        for algo_dir in sorted(scenario_dir.iterdir()):
            if not algo_dir.is_dir() or algo_dir.name == "ilp":
                continue
            runs = _run_dirs(algo_dir)
            if not runs:
                continue
            row = _row_from_run(scenario, algo_dir.name, runs[-1])
            if row:
                rows.append(row)
    return rows


@app.get("/api/results")
def get_results():
    return _collect_results()


@app.get("/api/history/{scenario}/{algo}")
def get_history(scenario: str, algo: str):
    algo_dir = RESULTS_ROOT / scenario / algo
    if not algo_dir.is_dir():
        raise HTTPException(404)
    rows = [_row_from_run(scenario, algo, d) for d in _run_dirs(algo_dir)]
    return [r for r in rows if r]


@app.get("/api/results/{scenario}/{algo}/{run}/{filename}")
def get_result_file(scenario: str, algo: str, run: str, filename: str):
    if filename not in ("training_curve.png", "comparison.png"):
        raise HTTPException(404)
    path = RESULTS_ROOT / scenario / algo / run / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


LOG_SOURCES_PATH = APP_DIR / "log_sources.json"
PHASE_RE = re.compile(r"===\s+\[(\w+)\]\s+(starting|finished)\s+(\w+)\s+at\s+(.+?)\s+===")


def _read_live_progress(scenario: str, algo: str) -> dict | None:
    """Read results/<scenario>/<algo>/.progress.json, written directly by the
    training callback (shared.paths.write_progress) on every progress tick.
    Not scraped from stdout: stdout is block-buffered whenever it's piped to
    a file instead of a TTY, so log-tailing for live progress lagged for
    minutes at a time — this file is overwritten fresh on every tick instead."""
    path = RESULTS_ROOT / scenario / algo / ".progress.json"
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

# Order the sequential shell loop launches algorithms in, per scenario — used to
# turn "which algo is currently running" into "N/6 models done" overall progress.
ALGO_ORDER = ["ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt", "dqn", "ddqn"]


def _load_log_sources() -> dict:
    """scenario -> log file path. Best-effort: this file just points at wherever
    each scenario's sequential-run stdout is being captured; update it any time
    the launch method changes. Missing/stale entries just mean no live progress."""
    try:
        return json.loads(LOG_SOURCES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _tail_log_progress(path_str: str) -> dict:
    path = Path(path_str)
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return {}

    completed_algos = []
    for line in lines:
        m = PHASE_RE.search(line)
        if not m:
            continue
        _, event, algo, _ = m.groups()
        if event == "finished" and algo not in completed_algos:
            completed_algos.append(algo)

    return {
        "completed_algos":   completed_algos,
        "log_tail":          lines[-12:],
    }


def _ps_snapshot() -> list[dict]:
    """Read-only process table lookup — never touches the processes themselves."""
    out = subprocess.run(
        ["ps", "-eo", "pid,psr,etimes,pcpu,cmd", "--no-headers"],
        capture_output=True, text=True, timeout=5,
    ).stdout
    procs = []
    for line in out.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, psr, etimes, pcpu, cmd = parts
        if re.search(r"scenarios/\w+/\w+/run_all\.py", cmd):
            procs.append({"pid": int(pid), "psr": int(psr), "etimes": int(etimes),
                          "pcpu": float(pcpu), "cmd": cmd})
    return procs


def _match_scenario_algo(cmd: str):
    m = re.search(r"scenarios/(\w+)/(\w+)/run_all\.py", cmd)
    return (m.group(1), m.group(2)) if m else (None, None)


@app.get("/api/progress")
def get_progress():
    procs = _ps_snapshot()
    log_sources = _load_log_sources()
    result = {}
    for scenario in SCENARIOS:
        proc = next((p for p in procs if _match_scenario_algo(p["cmd"])[0] == scenario), None)
        entry = {"scenario": scenario, "running": proc is not None}

        log_info = _tail_log_progress(log_sources.get(scenario, ""))
        completed_algos = log_info.pop("completed_algos", [])
        entry["completed_algos"] = completed_algos
        entry["log_tail"] = log_info.get("log_tail", [])

        current_algo = None
        latest_progress = None
        if proc:
            _, current_algo = _match_scenario_algo(proc["cmd"])
            entry.update({
                "current_algo":    current_algo,
                "pid":             proc["pid"],
                "core":            proc["psr"],
                "cpu_percent":     proc["pcpu"],
                "elapsed_seconds": proc["etimes"],
            })
            latest_progress = _read_live_progress(scenario, current_algo)
        entry["latest_progress"] = latest_progress

        total = len(ALGO_ORDER)
        done = len(completed_algos)
        within_current = (latest_progress["pct"] / 100.0) if (latest_progress and current_algo not in completed_algos) else 0.0
        entry["models_done"]  = done
        entry["models_total"] = total
        entry["overall_pct"]  = round(min(100.0, (done + within_current) / total * 100), 1)

        result[scenario] = entry
    return result


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = (APP_DIR / "static" / "index.html").read_text()
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
