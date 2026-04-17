#!/usr/bin/env bash

_copilot_all_run_main() {
	(
		set -euo pipefail

		# Use BASH_SOURCE so path resolution works for both "bash script.sh" and "source script.sh".
		local script_path root_dir python_bin cleanup_script
		local -a run_args cleanup_args
		local run_cleanup=0 cleanup_only=0 show_help=0
		script_path="${BASH_SOURCE[0]}"
		root_dir="$(cd "$(dirname "$script_path")/.." && pwd)"
		python_bin="$root_dir/.venv/bin/python"
		cleanup_script="$root_dir/run_scripts/cleanup_results.py"
		run_args=()
		cleanup_args=()

		while [[ $# -gt 0 ]]; do
			case "$1" in
				-h|--help)
					show_help=1
					shift
					;;
				--cleanup-results)
					run_cleanup=1
					shift
					;;
				--cleanup-only)
					run_cleanup=1
					cleanup_only=1
					shift
					;;
				--cleanup-today)
					run_cleanup=1
					cleanup_args+=(--today)
					shift
					;;
				--cleanup-incomplete)
					run_cleanup=1
					cleanup_args+=(--incomplete)
					shift
					;;
				--cleanup-dry-run)
					run_cleanup=1
					cleanup_args+=(--dry-run)
					shift
					;;
				--cleanup-date)
					run_cleanup=1
					if [[ $# -lt 2 ]]; then
						echo "Error: --cleanup-date requires a YYYYMMDD value." >&2
						exit 2
					fi
					cleanup_args+=(--date "$2")
					shift 2
					;;
				*)
					run_args+=("$1")
					shift
					;;
			esac
		done

		if [[ $show_help -eq 1 ]]; then
			echo "Usage: source run_scripts/all_run_shell_script.sh [run_all args] [cleanup options]"
			echo
			echo "Cleanup options:"
			echo "  --cleanup-results      Run cleanup after the training pipeline."
			echo "  --cleanup-only         Run cleanup only, skip training/copy/combined steps."
			echo "  --cleanup-today        Remove result directories from the target date."
			echo "  --cleanup-incomplete   Remove result directories missing core outputs."
			echo "  --cleanup-date DATE    Target date for --cleanup-today, format YYYYMMDD."
			echo "  --cleanup-dry-run      Preview cleanup targets without deleting them."
			echo
			echo "Run arguments are forwarded to run_scripts/run_all_problems.py:"
			"$python_bin" "$root_dir/run_scripts/run_all_problems.py" --help
			exit 0
		fi

		if [[ $cleanup_only -eq 0 ]]; then
			"$python_bin" -u "$root_dir/run_scripts/run_all_problems.py" "${run_args[@]}"
			"$python_bin" "$root_dir/run_scripts/copy_files.py"
			"$python_bin" "$root_dir/run_scripts/results_combined/combined.py"
		fi

		if [[ $run_cleanup -eq 1 ]]; then
			"$python_bin" "$cleanup_script" "${cleanup_args[@]}"
		fi
	)
}

_copilot_all_run_main "$@"
_copilot_all_run_status=$?
unset -f _copilot_all_run_main

return "$_copilot_all_run_status" 2>/dev/null || exit "$_copilot_all_run_status"