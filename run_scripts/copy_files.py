from pathlib import Path
from subprocess import run
import sys

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))
import timer_utils

print(f"Base path: {BASE_PATH}")


def auto_copy_file_to_results(source_path: Path, target_path: Path, source_filename: str, target_filename: str):
    """
    After each run script completes, automatically copy the specified output file to a common results directory with a standardized name.
    This helps in later aggregation and comparison of results across problems.
    auto choose the latest output folder name , folder name format: yyyymmdd_hhmmss
    """
    latest_folder = max((source_path / "results").iterdir(), key=lambda p: p.stat().st_mtime)
    source_path = source_path / "results" / latest_folder / source_filename
    target_path = target_path / target_filename
    
    if not source_path.exists():
        print(f"Warning: Expected output file {source_path} does not exist. Skipping copy.")
        return
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.rename(target_path)
    print(f"Copied {source_path} to {target_path}")



if __name__ == "__main__":
    print("Starting all problem runs...")
    # copy
    source_path = [
        BASE_PATH / "problem3_ppo",
        BASE_PATH / "problem4_ppo_mask",
        BASE_PATH / "problem5_ppo_lagrangian",
        BASE_PATH / "problem6_ppo_opt",
        BASE_PATH / "dqn",
    ]

    target_path = [
        BASE_PATH / "run_scripts" / "results_combined" / "problem3_ppo",
        BASE_PATH / "run_scripts" / "results_combined" / "problem4_ppo_mask",
        BASE_PATH / "run_scripts" / "results_combined" / "problem5_ppo_lagrangian",
        BASE_PATH / "run_scripts" / "results_combined" / "problem6_ppo_opt",
        BASE_PATH / "run_scripts" / "results_combined" / "dqn",
    ]

    for src, tgt in zip(source_path, target_path):
        auto_copy_file_to_results(src, tgt, "summary.csv", "summary.csv")
    
    # end copy
    print("\nAll files copied to results_combined. You can now run combined.py to aggregate and plot the results.")
