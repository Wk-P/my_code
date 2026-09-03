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
  GET  /api/batches                                    auto-discovered ad-hoc batch names under scripts/logs/
  GET  /api/batch_progress/{batch_name}                per-run status/results for one ad-hoc batch
  GET  /api/branch                                    current git branch + whether it has BC support
  GET  /api/system                                    CPU/load info, grouped by pinned core range
  GET  /                                              single-page dashboard (vanilla JS, no build step)
"""

import json
import os
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


def _git_tag_dates() -> dict[str, str]:
    try:
        r = subprocess.run(
            ["git", "for-each-ref", "--sort=-creatordate",
             "--format=%(refname:short)|%(creatordate:short)", "refs/tags"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        out = {}
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "|" in line:
                    tag, date = line.split("|", 1)
                    out[tag] = date
        return out
    except (OSError, subprocess.SubprocessError):
        return {}


# VERSION.md table rows look like: "| v1.2.3 | 2026-07-16 | 摘要文字 | [doc.md](doc.md) |"
_VERSION_TABLE_ROW = re.compile(
    r"^\|\s*(v[\d.]+)\s*\|\s*([\d-]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$"
)


def _parse_version_md() -> dict[str, dict]:
    """Reads version/VERSION.md's index table for a one-line summary (and,
    if present, a linked vX.Y.Z.md doc) per tag — this is the single source
    both this endpoint and humans editing the changelog read from, so the
    dashboard never drifts out of sync with what's written there."""
    path = PROJECT_ROOT / "version" / "VERSION.md"
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _VERSION_TABLE_ROW.match(line)
        if not m:
            continue
        tag, date, summary, doc_cell = m.groups()
        if tag == "版本":  # header row
            continue
        doc_m = re.search(r"\(([^)]+\.md)\)", doc_cell)
        out[tag] = {
            "date": date,
            "summary": summary,
            "doc_file": doc_m.group(1) if doc_m else None,
        }
    return out


@app.get("/api/tags")
def get_tags():
    """All git tags with a description for the frontend: VERSION.md's summary
    line, plus whether a standalone version/vX.Y.Z.md doc exists (fetch its
    body via /api/tags/{tag}/doc). Doesn't require a `git checkout` — reads
    tag metadata and the changelog file as committed on the current branch."""
    dates = _git_tag_dates()
    parsed = _parse_version_md()
    docs_dir = PROJECT_ROOT / "version"
    tags = []
    for tag in sorted(dates, key=lambda t: [int(x) for x in t.lstrip("v").split(".")], reverse=True):
        info = parsed.get(tag, {})
        doc_file = info.get("doc_file") or f"{tag}.md"
        has_doc = (docs_dir / doc_file).is_file() if doc_file else False
        tags.append({
            "tag":       tag,
            "date":      info.get("date") or dates.get(tag),
            "summary":   info.get("summary") or "",
            "has_doc":   has_doc,
            "doc_file":  doc_file if has_doc else None,
        })
    return tags


@app.get("/api/tags/{tag}/doc")
def get_tag_doc(tag: str):
    info = _parse_version_md().get(tag)
    doc_file = (info or {}).get("doc_file") or f"{tag}.md"
    path = PROJECT_ROOT / "version" / doc_file
    # version/ is a fixed, non-user-supplied directory and doc_file must
    # resolve inside it — reject anything that would climb out via "..".
    if ".." in Path(doc_file).parts or not path.is_file():
        raise HTTPException(404)
    return {"tag": tag, "doc_file": doc_file, "content": path.read_text(encoding="utf-8")}


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


MONITOR_STATE_PATH = APP_DIR / "monitor_state.json"


def _read_monitor_state() -> dict:
    """Written by app/backend/monitor.py (a separate long-running watchdog
    process, typically under systemd — see my-code-monitor.service) every
    ~30s. Missing file just means the watchdog isn't running yet; callers
    treat that the same as "no stuck/idle info available" rather than
    erroring, so /api/progress keeps working even before the watchdog is
    installed."""
    try:
        return json.loads(MONITOR_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


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


def _proc_stdout_log_path(pid: int) -> str | None:
    """Resolve the regular file a running process's stdout (fd 1) is
    redirected to, straight from /proc — independent of log_sources.json.
    Returns None if fd 1 isn't a regular file (TTY, pipe, closed, or the
    process/permission lookup fails), so callers can fall back to the
    static log_sources.json pointer for historical/no-longer-running runs."""
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None
    if not target.startswith("/") or not Path(target).is_file():
        return None
    return target


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
        if re.search(r"(?:^|[\s/])\w+/run_all(_bc)?\.py", cmd) or "self_imitation_finetune_v2.py" in cmd:
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

    # scripts/self_imitation_finetune_v2.py (see version/v2.1.0.md) is a
    # lt/ppo_mask-only side experiment, not part of the standard run_all(_bc)
    # pipeline — hardcoded scenario/algo since the script itself is
    # hardcoded to lt/ppo_mask (imports scenarios/lt/ppo_mask/config.py).
    # Writes its own results/<branch>/lt/ppo_mask_selfimit/.progress.json,
    # distinct from lt/ppo_mask's, so it can't collide with the real
    # training run's progress display.
    if re.search(r"self_imitation_finetune_v2\.py", cmd):
        return ("lt", "ppo_mask_selfimit")

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
    monitor_state = _read_monitor_state()
    result = {}
    for scenario in SCENARIOS:
        proc = next((p for p in procs if _match_scenario_algo(p["cmd"], p["pid"])[0] == scenario), None)
        entry = {"scenario": scenario, "running": proc is not None}

        # Prefer the actually-running process's own stdout target (works no
        # matter how it was launched) over the log_sources.json pointer,
        # which only gets updated by resume_scenario.sh/start_experiment.sh
        # and otherwise goes stale — showing a previous run's log forever.
        live_log_path = _proc_stdout_log_path(proc["pid"]) if proc else None
        log_info = _tail_log_progress(live_log_path or log_sources.get(scenario, ""))
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

        # "running": true just means a matching process exists — it says
        # nothing about whether it's actually making progress. monitor.py
        # tracks step-count movement across polls (this endpoint is
        # stateless per-request, so it can't) and flags stalls itself.
        mon = monitor_state.get(scenario)
        entry["monitor_status"]  = (mon or {}).get("status", "unknown")
        entry["stalled_seconds"] = (mon or {}).get("stalled_seconds")

        result[scenario] = entry
    return result


BATCH_LOG_DIR_RE = re.compile(
    r"^(?P<scenario>eq|gt|lt)_(?P<algo>\w+)_(?P<steps>\d+)_seed(?P<seed>\d+)_(?P<exp_id>[0-9a-f]+)\.log$"
)


@app.get("/api/batches")
def list_batches():
    """Auto-discovers every ad-hoc batch under scripts/logs/ so the frontend
    doesn't need a hardcoded, ever-growing list of batch names — any
    subdirectory containing at least one log file matching
    BATCH_LOG_DIR_RE counts as a batch. Sorted by most-recently-modified log
    file first (newest/most relevant batch on top)."""
    logs_root = PROJECT_ROOT / "scripts" / "logs"
    if not logs_root.is_dir():
        return {"batches": []}
    batches = []
    for d in logs_root.iterdir():
        if not d.is_dir():
            continue
        log_files = [f for f in d.glob("*.log") if BATCH_LOG_DIR_RE.match(f.name)]
        if not log_files:
            continue
        latest_mtime = max(f.stat().st_mtime for f in log_files)
        batches.append({"batch_name": d.name, "last_updated": latest_mtime, "run_count": len(log_files)})
    batches.sort(key=lambda b: b["last_updated"], reverse=True)
    return {"batches": batches}


@app.get("/api/batch_progress/{batch_name}")
def get_batch_progress(batch_name: str):
    """Progress view for an ad-hoc parallel batch launched by a script like
    scripts/<batch_name>.sh (e.g. eq_gt_migration_5seed), whose runs don't fit
    /api/progress's "one sequential process per scenario" assumption: many
    algos x many seeds run concurrently, each with its own exp_id, so
    "N/6 models done" is meaningless here — this counts run completion
    (results.json written) per (scenario, algo, seed) instead, which is
    unambiguous regardless of how many processes are running at once.

    Reads scripts/logs/<batch_name>/*.log filenames (written by the launch
    script itself, see e.g. scripts/eq_gt_migration_5seed.sh) as the source
    of truth for "which runs belong to this batch" — no manifest file
    required, so this stays accurate even mid-launch while new runs are
    still being started."""
    log_dir = PROJECT_ROOT / "scripts" / "logs" / batch_name
    if not log_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"no such batch log dir: {log_dir}")

    # Elapsed time since the batch's first run was launched, taken from the
    # oldest log file's ctime (each run's log is created the moment its
    # process is spawned — see e.g. scripts/eq_gt_migration_5seed.sh's
    # launch()). st_ctime on Linux is "inode change time", which for a
    # freshly-created file is its creation time — good enough here since
    # these log files are never touched again after creation.
    log_ctimes = [f.stat().st_ctime for f in log_dir.glob("*.log")]
    batch_started_at = min(log_ctimes) if log_ctimes else None
    elapsed_seconds = (time.time() - batch_started_at) if batch_started_at else None

    procs = _ps_snapshot()
    live_scenario_algo = {_match_scenario_algo(p["cmd"], p["pid"]) for p in procs}
    results_root = _results_root()

    runs = []
    for log_file in sorted(log_dir.glob("*.log")):
        m = BATCH_LOG_DIR_RE.match(log_file.name)
        if not m:
            continue
        scenario, algo, seed, exp_id = m["scenario"], m["algo"], m["seed"], m["exp_id"]
        run_dir = results_root / scenario / algo / exp_id
        results_path = run_dir / "results.json"
        done = results_path.is_file()
        running = (not done) and (scenario, algo) in live_scenario_algo
        run = {
            "scenario": scenario, "algo": algo, "seed": int(seed), "exp_id": exp_id,
            "status": "done" if done else ("running" if running else "queued"),
        }
        if done:
            # results.json nests the method's metrics under a method-specific
            # key that varies per algo (e.g. "ppo", "lagrange_ppo",
            # "maskable_ppo", "dqn", "ddqn") — pull whichever sub-dict has
            # ar_mean instead of hardcoding one key per algo.
            try:
                payload = json.loads(results_path.read_text())
                metrics = next(
                    (v for v in payload.values() if isinstance(v, dict) and "ar_mean" in v),
                    None,
                )
                if metrics:
                    run["ar_mean"] = metrics.get("ar_mean")
                    run["success_rate"] = metrics.get("success_rate")
            except (json.JSONDecodeError, OSError):
                pass
        runs.append(run)

    by_scenario: dict[str, dict] = {}
    for r in runs:
        sc = by_scenario.setdefault(r["scenario"], {"runs": [], "done": 0, "running": 0, "queued": 0})
        sc["runs"].append(r)
        sc[r["status"]] += 1

    total_done = sum(1 for r in runs if r["status"] == "done")
    return {
        "batch_name": batch_name,
        "total_runs": len(runs),
        "done": total_done,
        "overall_pct": round(100.0 * total_done / len(runs), 1) if runs else 0.0,
        "elapsed_seconds": round(elapsed_seconds, 1) if elapsed_seconds is not None else None,
        "by_scenario": by_scenario,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = (APP_DIR / "static" / "index.html").read_text()
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
