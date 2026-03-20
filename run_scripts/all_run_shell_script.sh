#!/usr/bin/env bash

_copilot_all_run_main() {
	(
		set -euo pipefail

		# Use BASH_SOURCE so path resolution works for both "bash script.sh" and "source script.sh".
		local script_path root_dir python_bin
		script_path="${BASH_SOURCE[0]}"
		root_dir="$(cd "$(dirname "$script_path")/.." && pwd)"
		python_bin="$root_dir/.venv/bin/python"

		for arg in "$@"; do
			if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
				"$python_bin" "$root_dir/run_scripts/run_all_problems.py" "$@"
				exit 0
			fi
		done

		"$python_bin" "$root_dir/run_scripts/run_all_problems.py" "$@"
		"$python_bin" "$root_dir/run_scripts/copy_files.py"
		"$python_bin" "$root_dir/run_scripts/results_combined/combined.py"
	)
}

_copilot_all_run_main "$@"
_copilot_all_run_status=$?
unset -f _copilot_all_run_main

return "$_copilot_all_run_status" 2>/dev/null || exit "$_copilot_all_run_status"