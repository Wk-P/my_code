from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parent.parent
RESULT_PATHS = [
    BASE_PATH / "problem3_ppo" / "results",
    BASE_PATH / "problem4_ppo_mask" / "results",
    BASE_PATH / "problem5_ppo_lagrangian" / "results",
    BASE_PATH / "problem6_ppo_opt" / "results",
    BASE_PATH / "dqn" / "results",
]
REQUIRED_FILES = ["results.json", "summary.csv", "training_curve.png", "comparison.png"]
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean result directories by removing today's runs and/or incomplete records.",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Delete result directories created on the target date.",
    )
    parser.add_argument(
        "--incomplete",
        action="store_true",
        help="Delete result directories missing any required output file.",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y%m%d"),
        help="Target date used with --today, format: YYYYMMDD. Defaults to today's date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing anything.",
    )
    args = parser.parse_args()
    if not args.today and not args.incomplete:
        args.today = True
        args.incomplete = True
    return args


def iter_result_dirs() -> list[Path]:
    result_dirs: list[Path] = []
    for result_path in RESULT_PATHS:
        if not result_path.exists():
            continue
        for child in sorted(result_path.iterdir()):
            if child.is_dir() and TIMESTAMP_DIR_RE.match(child.name):
                result_dirs.append(child)
    return result_dirs


def get_missing_files(result_dir: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (result_dir / name).exists()]


def collect_candidates(delete_today: bool, delete_incomplete: bool, date_prefix: str) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []
    for result_dir in iter_result_dirs():
        reasons: list[str] = []
        if delete_today and result_dir.name.startswith(f"{date_prefix}_"):
            reasons.append(f"today:{date_prefix}")
        if delete_incomplete:
            missing_files = get_missing_files(result_dir)
            if missing_files:
                reasons.append("missing:" + ",".join(missing_files))
        if reasons:
            candidates.append(CleanupCandidate(path=result_dir, reasons=tuple(reasons)))
    return candidates


def remove_candidates(candidates: list[CleanupCandidate], dry_run: bool) -> None:
    action = "Would remove" if dry_run else "Removing"
    for candidate in candidates:
        reason_text = "; ".join(candidate.reasons)
        print(f"{action}: {candidate.path.relative_to(BASE_PATH)} [{reason_text}]")
        if not dry_run:
            shutil.rmtree(candidate.path)


def main() -> int:
    args = parse_args()
    candidates = collect_candidates(
        delete_today=args.today,
        delete_incomplete=args.incomplete,
        date_prefix=args.date,
    )

    if not candidates:
        print("No matching result directories found.")
        return 0

    remove_candidates(candidates, dry_run=args.dry_run)
    print(f"Matched {len(candidates)} result directories.")
    if args.dry_run:
        print("Dry run only. No files were removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())