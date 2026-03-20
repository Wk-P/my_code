# run all problems run_all.py (problem3, problem4, problem5, problem6, dqn)

from pathlib import Path
import argparse
from subprocess import PIPE, STDOUT, CalledProcessError, Popen
import os
import sys
import time

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))
import timer_utils

RUN_LIST = [
    "problem3_ppo/run_all.py",
    "problem4_ppo_mask/run_all.py",
    "problem5_ppo_lagrangian/run_all.py",
    "problem6_ppo_opt/run_all.py",
    "dqn/run_all.py",
]

PYTHON_PATH = BASE_PATH / ".venv" / "bin" / "python"


def parse_args():
    parser = argparse.ArgumentParser(description="Run all problem run_all.py scripts.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps for all child run_all.py scripts.",
    )
    return parser.parse_args()


@timer_utils.timer
def main(total_timesteps: int | None = None):
    print(f"Base path: {BASE_PATH}")
    for run_script in RUN_LIST:
        print(f"Scheduled run: {run_script}")
    print("=== PPO/DQN Training & Evaluation for All Problems ===")
    print(f"Using Python interpreter at: {PYTHON_PATH}")

    if not PYTHON_PATH.exists():
        print(
            f"Warning: {PYTHON_PATH} does not exist. "
            "Make sure to set up the virtual environment and install dependencies."
        )
        return 1

    print("All run scripts and Python interpreter found.\n")
    print("Checking run script existence...")
    for run_script in RUN_LIST:
        run_path = BASE_PATH / run_script
        if not run_path.exists():
            print(f"Error: {run_path} does not exist. Please check the RUN_LIST.")
            return 1
    print("All run scripts exist. Starting execution...\n")

    for run_script in RUN_LIST:
        print(f"\n=== Running {run_script} ===\n", flush=True)
        # Use unbuffered child stdout so progress lines are visible immediately.
        cmd = [str(PYTHON_PATH), "-u", str(BASE_PATH / run_script)]
        if total_timesteps is not None:
            cmd.extend(["--total-timesteps", str(total_timesteps)])

        t0 = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = Popen(
            cmd,
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{run_script}] {line}", end="")

        rc = proc.wait()
        elapsed = time.time() - t0
        if rc != 0:
            print(f"Error: {run_script} failed with exit code {rc}.")
            return rc
        print(f"=== Finished {run_script} in {elapsed:.1f}s ===", flush=True)

    return 0


if __name__ == "__main__":
    args = parse_args()
    print("Starting all problem runs...")
    if args.total_timesteps is not None:
        print(f"[override] total_timesteps={args.total_timesteps}")
    rc = main(total_timesteps=args.total_timesteps)
    if rc == 0:
        print("All runs completed.")
    raise SystemExit(rc)