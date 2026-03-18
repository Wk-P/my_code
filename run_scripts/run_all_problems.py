# run all problems run_all.py (problem3, problem4, problem5, problem6, dqn)

from pathlib import Path
from subprocess import run
import sys

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
        print(f"\n=== Running {run_script} ===\n")
        result = run([str(PYTHON_PATH), str(BASE_PATH / run_script)], check=True, capture_output=True)

        output = result.stdout.decode() if result.stdout else ""
        error = result.stderr.decode() if result.stderr else ""

        if output:
            print(f"Output from {run_script}:\n{output}")
        if error:
            print(f"Error from {run_script}:\n{error}")

if __name__ == "__main__":
    print("Starting all problem runs...")
    main()
    print("All runs completed.")