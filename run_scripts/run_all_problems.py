# run all problems run_all.py (problem3, problem4, problem5, problem6, dqn)

from pathlib import Path
from subprocess import PIPE, STDOUT, CalledProcessError, Popen
import os
import sys
import time

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))
import timer_utils

print(f"Base path: {BASE_PATH}")

RUN_LIST = [
    "problem3_ppo/run_all.py",
    "problem4_ppo_mask/run_all.py",
    "problem5_ppo_lagrangian/run_all.py",
    "problem6_ppo_opt/run_all.py",
    "dqn/run_all.py",
]

print(f"Base path: {BASE_PATH}")
for run_script in RUN_LIST:
    print(f"Scheduled run: {run_script}")
print("=== PPO/DQN Training & Evaluation for All Problems ===")
PYTHON_PATH = BASE_PATH / ".venv" / "bin" / "python"
print(f"Using Python interpreter at: {PYTHON_PATH}")
if not PYTHON_PATH.exists():
    print(f"Warning: {PYTHON_PATH} does not exist. Make sure to set up the virtual environment and install dependencies.")
    exit(1)
print("All run scripts and Python interpreter found.\n")

# check run list existence
print("Checking run script existence...")
for run_script in RUN_LIST:
    run_path = BASE_PATH / run_script
    if not run_path.exists():
        print(f"Error: {run_path} does not exist. Please check the RUN_LIST.")
        exit(1)
print("All run scripts exist. Starting execution...\n")




@timer_utils.timer
def main():
    for run_script in RUN_LIST:
        print(f"\n=== Running {run_script} ===\n", flush=True)
        # Use unbuffered child stdout so progress lines are visible immediately.
        cmd = [str(PYTHON_PATH), "-u", str(BASE_PATH / run_script)]
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
            raise CalledProcessError(rc, cmd)
        print(f"=== Finished {run_script} in {elapsed:.1f}s ===", flush=True)

if __name__ == "__main__":
    print("Starting all problem runs...")
    main()
    print("All runs completed.")