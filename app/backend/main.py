"""
my-code experiment dashboard — read-only view over
results/<branch>/<scenario>/<algo>/<run>/ for the currently checked-out git
branch, plus best-effort process/log introspection for live training
progress.

This module never imports or executes anything under scenarios/ or shared/ —
it only reads files (results.json, PNGs, log files) and OS process state
(ps, /proc, git). It cannot affect running or future training runs.

Serves:
  GET  /api/results                                  summary table (latest run per scenario/algo)
  GET  /api/results/{scenario}/{algo}/{run}/{file}    training_curve.png / comparison.png
  GET  /api/history/{scenario}/{algo}                 all historical runs for one algo
  GET  /api/progress                                  live training progress per scenario
  GET  /api/branch                                    current git branch + whether it has BC support
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

APP_DIR           = Path(__file__).parent
PROJECT_ROOT      = APP_DIR.parent.parent
RESULTS_ROOT_BASE = PROJECT_ROOT / "results"
SCENARIOS         = ["eq", "gt", "lt"]

app = FastAPI(title="my-code experiment dashboard")
app.mount("/assets", StaticFiles(directory=APP_DIR / "static" / "assets"), name="assets")


# ── Git branch awareness ────────────────────────────────────────────────────
#
# results/<branch>/... mirrors shared/paths.py's own branch-scoped
# results_dir() — see that module's docstring. This backend is a long-lived
# server (unlike the one-shot training scripts), so the branch has to be
# re-resolved on every request rather than cached at import time: a `git
# checkout` in another terminal must show up here without restarting uvicorn.

def _git_current_branch() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_branch_list() -> list[str]:
    try:
        r = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads/"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return [b for b in r.stdout.splitlines() if b] if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


def _results_root(branch: str | None = None) -> Path:
    """results/<branch>/ — defaults to the currently checked-out branch, but
    every data endpoint accepts an explicit `?branch=` query param so the
    dashboard can browse any branch's already-recorded results without an
    actual `git checkout` (which would disrupt whatever's in the working
    tree). Re-resolves the current branch on every call — see module note
    above — rather than caching it, so a `git checkout` elsewhere is picked
    up without restarting uvicorn. Falls back to the current branch (not the
    literal string) if the requested branch doesn't exist, so a stale/typo'd
    `?branch=` can't silently point at a nonexistent directory."""
    current = _git_current_branch() or "unknown"
    if branch and branch in _git_branch_list():
        return RESULTS_ROOT_BASE / branch
    return RESULTS_ROOT_BASE / current


@app.get("/api/branch")
def get_branch():
    return {
        "current":      _git_current_branch(),
        "branches":     _git_branch_list(),
        "bc_supported": (PROJECT_ROOT / "shared" / "bc_pretrain.py").is_file(),
    }


def _algo_key(data: dict) -> str | None:
    # Any results.json top-level key that isn't the algo's own result block
    # must be listed here, or _algo_key() mistakes it for one and every
    # .get("ar_mean")-style read below blows up on a str/int instead of the
    # algo eval dict. "bc" and "exp_id" are both metadata added by
    # run_all_bc.py — see shared/bc_pretrain.py.
    reserved = ("scenario", "prototype_scenario", "scenario_count",
                "train_count", "test_count", "N", "M", "ilp", "training",
                "feasibility", "created_at", "bc", "exp_id")
    return next((k for k in data.keys() if k not in reserved), None)


def _row_from_run(scenario: str, algo: str, run_dir: Path) -> dict | None:
    try:
        data = json.loads((run_dir / "results.json").read_text())
    except (json.JSONDecodeError, OSError):
        return None
    algo_key  = _algo_key(data)
    algo_eval = data.get(algo_key, {}) if algo_key else {}
    training  = data.get("training", {})
    ilp       = data.get("ilp", {})
    # run_all_bc.py (ILP behavior-cloning pretrain) writes its result key as
    # "<algo>_bc" (e.g. "maskable_ppo_bc") — see shared/bc_pretrain.py and
    # each scenarios/<scenario>/<algo>/run_all_bc.py. That's the single
    # source of truth for "is this a BC run", since it comes straight from
    # the training script itself rather than a directory-naming convention.
    is_bc = bool(algo_key) and algo_key.endswith("_bc")
    # "run" (run_dir.name) and "exp_id" (the whole eq+gt+lt batch id, see
    # shared/paths.new_exp_id) are different things that happened to always
    # be equal — until run_all_bc.py started suffixing the run dir with
    # "_bc" to keep it separate from the baseline run in the same algo
    # folder. run_all_bc.py now writes "exp_id" into results.json explicitly
    # so batch grouping (ExperimentTree) can use the real id instead of the
    # directory name. Older baseline runs never had a reason to diverge, so
    # falling back to run_dir.name for them is exact, not approximate.
    exp_id = data.get("exp_id") or run_dir.name
    return {
        "scenario":      scenario,
        "algo":          algo,
        "display_algo":  f"{algo}+bc" if is_bc else algo,
        "is_bc":         is_bc,
        "run":           run_dir.name,
        "exp_id":        exp_id,
        "created_at":    data.get("created_at"),
        "N":             data.get("N"),
        "M":             data.get("M"),
        "train_count":   data.get("train_count"),
        "test_count":    data.get("test_count"),
        "ilp_ar":        ilp.get("ar"),
        "test_ar_mean":  algo_eval.get("ar_mean"),
        "test_ar_std":   algo_eval.get("ar_std"),
        "test_success_rate":        algo_eval.get("success_rate"),
        "test_cap_viol_rate":       algo_eval.get("cap_viol_rate"),
        "test_conflict_viol_rate":  algo_eval.get("conflict_viol_rate"),
        "test_cap_viol_total":      algo_eval.get("cap_viol_total"),
        "test_conflict_viol_total": algo_eval.get("conflict_viol_total"),
        "train_ar_last50": training.get("ar_last50"),
        "train_steps":     training.get("total_steps"),
        "train_episodes":  training.get("n_episodes"),
    }


_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")


def _sort_key(run_dir: Path) -> str:
    """Run dirs used to be named as timestamps (lexicographic == chronological);
    now they're random exp_id hashes, so chronology has to come from
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


def _collect_results(branch: str | None = None) -> list[dict]:
    rows = []
    results_root = _results_root(branch)
    for scenario in SCENARIOS:
        scenario_dir = results_root / scenario
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


def _collect_all_experiments(branch: str | None = None) -> list[dict]:
    """Every historical run across every scenario/algo, not just the latest
    one per algo — powers the EXP_ID -> scenario -> algo tree, which needs
    the full history to group by batch (exp_id)."""
    rows = []
    results_root = _results_root(branch)
    for scenario in SCENARIOS:
        scenario_dir = results_root / scenario
        if not scenario_dir.is_dir():
            continue
        for algo_dir in sorted(scenario_dir.iterdir()):
            if not algo_dir.is_dir() or algo_dir.name == "ilp":
                continue
            for run_dir in _run_dirs(algo_dir):
                row = _row_from_run(scenario, algo_dir.name, run_dir)
                if row:
                    rows.append(row)
    return rows


@app.get("/api/results")
def get_results(branch: str | None = None):
    return _collect_results(branch)


@app.get("/api/experiments")
def get_experiments(branch: str | None = None):
    return _collect_all_experiments(branch)


@app.get("/api/history/{scenario}/{algo}")
def get_history(scenario: str, algo: str, branch: str | None = None):
    algo_dir = _results_root(branch) / scenario / algo
    if not algo_dir.is_dir():
        raise HTTPException(404)
    rows = [_row_from_run(scenario, algo, d) for d in _run_dirs(algo_dir)]
    return [r for r in rows if r]


@app.get("/api/results/{scenario}/{algo}/{run}/{filename}")
def get_result_file(scenario: str, algo: str, run: str, filename: str, branch: str | None = None):
    if filename not in ("training_curve.png", "comparison.png"):
        raise HTTPException(404)
    path = _results_root(branch) / scenario / algo / run / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


LOG_SOURCES_PATH = APP_DIR / "log_sources.json"
PHASE_RE = re.compile(r"===\s+\[(\w+)\]\s+(starting|finished)\s+(\w+)\s+at\s+(.+?)\s+===")


def _read_live_progress(scenario: str, algo: str) -> dict | None:
    """Read results/<branch>/<scenario>/<algo>/.progress.json, written
    directly by the training callback (shared.paths.write_progress) on every
    progress tick. Not scraped from stdout: stdout is block-buffered whenever
    it's piped to a file instead of a TTY, so log-tailing for live progress
    lagged for minutes at a time — this file is overwritten fresh on every
    tick instead.

    KNOWN LIMITATION: run_all.py and run_all_bc.py share the same C.OUTDIR
    (and thus the same .progress.json path) for a given branch/scenario/algo,
    since the BC variant only diverges at the run-directory level (exp_id +
    "_bc"), not the algo dir. If baseline and BC training run concurrently
    for the same algo on the same branch checkout, whichever writes last
    "wins" the progress file — this display can transiently show the wrong
    run's step count. Final results.json are unaffected (each run gets its
    own directory). Running them on different branches no longer collides,
    since results/<branch>/... physically separates them."""
    algo = algo.removesuffix("+bc")
    path = _results_root() / scenario / algo / ".progress.json"
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
        if re.search(r"(?:^|[\s/])\w+/run_all(_bc)?\.py", cmd):
            procs.append({"pid": int(pid), "psr": int(psr), "etimes": int(etimes),
                          "pcpu": float(pcpu), "cmd": cmd})
    return procs


def _match_scenario_algo(cmd: str, pid: int | None = None):
    """Returns (scenario, algo) where algo gets a "+bc" suffix when the
    process is running run_all_bc.py (ILP behavior-cloning pretrain variant),
    so it doesn't get conflated with the plain run_all.py baseline in the
    live-progress display.

    Two invocation shapes are supported:
      - the standard scripts/start_experiment.sh launcher, which always uses
        the absolute .../scenarios/<scenario>/<algo>/run_all.py form;
      - an ad-hoc `cd scenarios/<scenario> && python3 <algo>/run_all[_bc].py`
        invocation (e.g. a manually backgrounded comparison run), whose cmd
        only contains "<algo>/run_all.py" — the scenario is recovered from
        /proc/<pid>/cwd instead.
    """
    m = re.search(r"scenarios/(\w+)/(\w+)/run_all(_bc)?\.py", cmd)
    if m:
        scenario, algo, is_bc = m.group(1), m.group(2), m.group(3)
        return (scenario, f"{algo}+bc" if is_bc else algo)

    m = re.search(r"(?:^|[\s/])(\w+)/run_all(_bc)?\.py", cmd)
    if not m or pid is None:
        return (None, None)
    algo, is_bc = m.group(1), m.group(2)
    try:
        cwd = Path(f"/proc/{pid}/cwd").resolve()
    except OSError:
        return (None, None)
    scenario = cwd.name
    if scenario not in SCENARIOS:
        return (None, None)
    return (scenario, f"{algo}+bc" if is_bc else algo)


@app.get("/api/progress")
def get_progress():
    procs = _ps_snapshot()
    log_sources = _load_log_sources()
    result = {}
    for scenario in SCENARIOS:
        proc = next((p for p in procs if _match_scenario_algo(p["cmd"], p["pid"])[0] == scenario), None)
        entry = {"scenario": scenario, "running": proc is not None}

        log_info = _tail_log_progress(log_sources.get(scenario, ""))
        completed_algos = log_info.pop("completed_algos", [])
        entry["completed_algos"] = completed_algos
        entry["log_tail"] = log_info.get("log_tail", [])

        current_algo = None
        latest_progress = None
        if proc:
            _, current_algo = _match_scenario_algo(proc["cmd"], proc["pid"])
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
