# run all problems in PARALLEL

from pathlib import Path
import argparse
from subprocess import PIPE, STDOUT, Popen
import os
import sys
import time
import threading

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))
import timer_utils

RUN_LIST = [
    "problem3_ppo/run_all.py",
    "problem4_ppo_mask/run_all.py",
    "problem5_ppo_lagrangian/run_all.py",
    "problem6_ppo_opt/run_all.py",
    "problem_dqn/run_all.py",
    "problem_ddqn/run_all.py",
]

PYTHON_PATH = BASE_PATH.parent / ".venv" / "bin" / "python"


def parse_args():
    parser = argparse.ArgumentParser(description="Run all problems in parallel.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps for all child run_all.py scripts.",
    )
    return parser.parse_args()


def stream_output(run_script: str, proc: Popen, results: dict) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{run_script}] {line}", end="", flush=True)
    rc = proc.wait()
    results[run_script] = rc


@timer_utils.timer
def main(total_timesteps: int | None = None):
    print(f"Base path: {BASE_PATH}")
    print("=== Parallel PPO/DQN Training & Evaluation for All Problems ===")
    print(f"Using Python interpreter at: {PYTHON_PATH}")

    if not PYTHON_PATH.exists():
        print(f"Warning: {PYTHON_PATH} does not exist.")
        return 1

    for run_script in RUN_LIST:
        if not (BASE_PATH / run_script).exists():
            print(f"Error: {BASE_PATH / run_script} does not exist.")
            return 1

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    procs: dict[str, Popen] = {}
    results: dict[str, int] = {}

    # Launch all in parallel
    for run_script in RUN_LIST:
        cmd = [str(PYTHON_PATH), "-u", str(BASE_PATH / run_script)]
        if total_timesteps is not None:
            cmd.extend(["--total-timesteps", str(total_timesteps)])
        proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, env=env)
        procs[run_script] = proc
        print(f"Started: {run_script} (pid={proc.pid})", flush=True)

    # Stream output from each process in a separate thread
    threads = []
    for run_script, proc in procs.items():
        t = threading.Thread(
            target=stream_output, args=(run_script, proc, results), daemon=True
        )
        t.start()
        threads.append(t)

    t0 = time.time()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    print(f"\n=== All problems finished in {elapsed:.1f}s ===", flush=True)
    for run_script, rc in results.items():
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {run_script}: {status}")

    failed = [s for s, rc in results.items() if rc != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    args = parse_args()
    print("Starting all problems in parallel...")
    if args.total_timesteps is not None:
        print(f"[override] total_timesteps={args.total_timesteps}")
    rc = main(total_timesteps=args.total_timesteps)
    if rc == 0:
        print("All runs completed successfully.")
    raise SystemExit(rc)
