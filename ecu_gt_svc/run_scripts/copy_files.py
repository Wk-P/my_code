from pathlib import Path
from shutil import copy2
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
    results_dir = source_path / "results"
    timestamp_dirs = [
        path for path in results_dir.iterdir()
        if path.is_dir() and len(path.name) == 15 and path.name[8] == "_"
    ]

    if not timestamp_dirs:
        print(f"Warning: No timestamped result folders found under {results_dir}. Skipping copy.")
        return

    latest_folder = None
    for candidate in sorted(timestamp_dirs, key=lambda path: path.name, reverse=True):
        candidate_file = candidate / source_filename
        if candidate_file.exists():
            latest_folder = candidate
            source_path = candidate_file
            break

    if latest_folder is None:
        print(
            f"Warning: No timestamped result folder under {results_dir} contains {source_filename}. "
            "Skipping copy."
        )
        return

    target_path = target_path / target_filename
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(source_path, target_path)
    print(f"Copied {source_path} to {target_path}")



if __name__ == "__main__":
    print("Starting all problem runs...")
    # copy
    source_path = [
        BASE_PATH / "problem3_ppo",
        BASE_PATH / "problem4_ppo_mask",
        BASE_PATH / "problem5_ppo_lagrangian",
        BASE_PATH / "problem6_ppo_opt",
        BASE_PATH / "problem_dqn",
        BASE_PATH / "problem_ddqn"
    ]

    target_path = [
        BASE_PATH / "run_scripts" / "results_combined" / "problem3_ppo",
        BASE_PATH / "run_scripts" / "results_combined" / "problem4_ppo_mask",
        BASE_PATH / "run_scripts" / "results_combined" / "problem5_ppo_lagrangian",
        BASE_PATH / "run_scripts" / "results_combined" / "problem6_ppo_opt",
        BASE_PATH / "run_scripts" / "results_combined" / "problem_dqn",
        BASE_PATH / "run_scripts" / "results_combined" / "problem_ddqn"
    ]

    for src, tgt in zip(source_path, target_path):
        auto_copy_file_to_results(src, tgt, "summary.csv", "summary.csv")
    
    # end copy
    print("\nAll files copied to results_combined. You can now run combined.py to aggregate and plot the results.")
