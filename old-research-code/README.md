# Reinforcement Learning VS Optimization

This README focuses on the practical usage of the scripts in this repository, so you can directly run training, aggregation, and cleanup workflows.

Chinese version: see README_CN.md.

## 1. Project Structure

### Core Directories

<table>
  <thead>
    <tr>
      <th>Directory</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>problem1</td>
      <td>Base environment definitions and early problem setup</td>
    </tr>
    <tr>
      <td>problem2_ilp</td>
      <td>ILP config generation and optimal-solution solver</td>
    </tr>
    <tr>
      <td>problem3_ppo</td>
      <td>PPO without safety constraints</td>
    </tr>
    <tr>
      <td>problem4_ppo_mask</td>
      <td>PPO with action masking</td>
    </tr>
    <tr>
      <td>problem5_ppo_lagrangian</td>
      <td>Lagrangian constrained PPO</td>
    </tr>
    <tr>
      <td>problem6_ppo_opt</td>
      <td>PPO with patch optimization logic</td>
    </tr>
    <tr>
      <td>dqn</td>
      <td>DQN baseline</td>
    </tr>
    <tr>
      <td>run_scripts</td>
      <td>One-click run, result copy, combined plotting, and cleanup</td>
    </tr>
  </tbody>
</table>

### Result Directories

Each problem directory usually contains a results subdirectory. A single run creates one timestamped directory, for example:

- 20260321_071733

The standard outputs usually include:

- results.json
- summary.csv
- training_curve.png
- comparison.png

## 2. Environment Setup

The recommended Python interpreter is the virtual environment inside the repository:

    .venv/bin/python

If dependencies are not installed yet:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

On Linux, the recommended full path is:

    /home/soar009/github/my_code/.venv/bin/python

Default training steps for all trainable problems are now centralized in:

    training_steps_config.py

If you want to change the default TOTAL_STEPS for all problems, edit that file instead of editing each problem's config.py separately.

## 3. Main Entry Points

### 3.1 Run All Problems with One Command

Script: run_scripts/all_run_shell_script.sh

Purpose:

- Run P3, P4, P5, P6, and DQN in sequence
- Copy the latest summary.csv into the combined-results directory
- Generate combined_results.png automatically
- Optionally clean result directories after the run

Most common command:

    source run_scripts/all_run_shell_script.sh --total-timesteps 500000

Quick smoke test:

    source run_scripts/all_run_shell_script.sh --total-timesteps 1000

Run cleanup after training, including today's results and incomplete records:

    source run_scripts/all_run_shell_script.sh --total-timesteps 1000 --cleanup-results

Cleanup only, without training:

    source run_scripts/all_run_shell_script.sh --cleanup-only --cleanup-today --cleanup-date 20260321

Show help:

    bash run_scripts/all_run_shell_script.sh --help

Supported cleanup options:

- --cleanup-results
- --cleanup-only
- --cleanup-today
- --cleanup-incomplete
- --cleanup-date YYYYMMDD
- --cleanup-dry-run

### 3.2 Run All Problems with Python Only

Script: run_scripts/run_all_problems.py

Purpose:

- Only executes the five run_all.py scripts in sequence
- Does not copy summary.csv files
- Does not generate the combined figure

Command:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/run_all_problems.py

Quick smoke test:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/run_all_problems.py --total-timesteps 1000

Show help:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/run_all_problems.py --help

## 4. Run Each Problem Individually

Each of the following scripts creates a new timestamped directory under its own results folder.

### 4.1 P3: PPO Without Constraints

Script: problem3_ppo/run_all.py

Command:

    /home/soar009/github/my_code/.venv/bin/python problem3_ppo/run_all.py

Quick smoke test:

    /home/soar009/github/my_code/.venv/bin/python problem3_ppo/run_all.py --total-timesteps 1000

### 4.2 P4: Masked PPO

Script: problem4_ppo_mask/run_all.py

Command:

    /home/soar009/github/my_code/.venv/bin/python problem4_ppo_mask/run_all.py

Quick smoke test:

    /home/soar009/github/my_code/.venv/bin/python problem4_ppo_mask/run_all.py --total-timesteps 1000

### 4.3 P5: Lagrangian PPO

Script: problem5_ppo_lagrangian/run_all.py

Command:

    /home/soar009/github/my_code/.venv/bin/python problem5_ppo_lagrangian/run_all.py

Quick smoke test:

    /home/soar009/github/my_code/.venv/bin/python problem5_ppo_lagrangian/run_all.py --total-timesteps 1000

### 4.4 P6: PPO with Patch Optimization

Script: problem6_ppo_opt/run_all.py

Command:

    /home/soar009/github/my_code/.venv/bin/python problem6_ppo_opt/run_all.py

Quick smoke test:

    /home/soar009/github/my_code/.venv/bin/python problem6_ppo_opt/run_all.py --total-timesteps 1000

### 4.5 DQN

Script: dqn/run_all.py

Command:

    /home/soar009/github/my_code/.venv/bin/python dqn/run_all.py

Quick smoke test:

    /home/soar009/github/my_code/.venv/bin/python dqn/run_all.py --total-timesteps 1000

### 4.6 Shared Argument for All run_all.py Scripts

All five training entry points currently support:

- --total-timesteps TOTAL_TIMESTEPS

Purpose: override TOTAL_STEPS in the corresponding config for smoke tests and debugging.

Default values are loaded from the centralized file:

    training_steps_config.py

## 5. ILP Scripts

### 5.1 Generate ILP Configurations

Script: problem2_ilp/config/generate_config.py

Purpose:

- Generate multiple scenarios randomly
- Save them to problem2_ilp/config/config_YYYYMMDD_HHMMSS.yaml

Command:

    /home/soar009/github/my_code/.venv/bin/python problem2_ilp/config/generate_config.py

Notes:

- This script currently has no command-line arguments
- The number of scenarios is hard-coded in the script and defaults to 200

### 5.2 Solve ILP and Generate Summary Outputs

Script: problem2_ilp/optimal_solution/main.py

Purpose:

- Read a YAML config from problem2_ilp/config
- Solve all scenarios with ILP
- Generate shared cache files and summary plots

Command:

    /home/soar009/github/my_code/.venv/bin/python problem2_ilp/optimal_solution/main.py

Notes:

- This script currently has no standard command-line arguments
- The config file is controlled by the default config_filename near the bottom of the script
- To switch configs, you need to change that default filename in the script

## 6. Aggregation and Copy Scripts

### 6.1 Copy the Latest summary.csv into the Combined Directory

Script: run_scripts/copy_files.py

Purpose:

- Copy summary.csv from the latest valid result directory of each problem
- Save each one into the matching subdirectory under run_scripts/results_combined

Command:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/copy_files.py

Output files:

- run_scripts/results_combined/problem3_ppo/summary.csv
- run_scripts/results_combined/problem4_ppo_mask/summary.csv
- run_scripts/results_combined/problem5_ppo_lagrangian/summary.csv
- run_scripts/results_combined/problem6_ppo_opt/summary.csv
- run_scripts/results_combined/dqn/summary.csv

### 6.2 Generate the Combined Comparison Figure

Script: run_scripts/results_combined/combined.py

Purpose:

- Read the five summary.csv files from the combined-results directory
- Generate the combined comparison figure combined_results.png

Command:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/results_combined/combined.py

Output file:

- run_scripts/results_combined/combined_results.png

The current combined figure shows three categories of information:

- AR comparison
- Violation-rate comparison
- Placement-completeness comparison

## 7. Cleanup Script

Script: run_scripts/cleanup_results.py

Purpose:

- Delete result directories generated on a target date
- Delete incomplete result records

Command examples:

Preview what would be deleted:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --dry-run

Delete only today's results:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --today

Delete only incomplete records:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --incomplete

Delete results from a specific date:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --today --date 20260321

Delete both date-matched results and incomplete records:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/cleanup_results.py --today --incomplete --date 20260321

If neither --today nor --incomplete is provided:

- The script enables both behaviors by default

## 8. Recommended Workflows

### Workflow A: Simplest End-to-End Path

Recommended for regular training and plotting:

    source run_scripts/all_run_shell_script.sh --total-timesteps 500000

### Workflow B: Debug One Method First

Recommended when tuning a single method:

    /home/soar009/github/my_code/.venv/bin/python problem5_ppo_lagrangian/run_all.py --total-timesteps 1000

Then refresh the combined outputs with:

    /home/soar009/github/my_code/.venv/bin/python run_scripts/copy_files.py
    /home/soar009/github/my_code/.venv/bin/python run_scripts/results_combined/combined.py

### Workflow C: Train and Cleanup

    source run_scripts/all_run_shell_script.sh --total-timesteps 1000 --cleanup-results --cleanup-today --cleanup-incomplete

## 9. Common Output Locations

### Per-Problem Outputs

- problem3_ppo/results/<timestamped_directory>
- problem4_ppo_mask/results/<timestamped_directory>
- problem5_ppo_lagrangian/results/<timestamped_directory>
- problem6_ppo_opt/results/<timestamped_directory>
- dqn/results/<timestamped_directory>

### Combined Outputs

- run_scripts/results_combined/problem\*/summary.csv
- run_scripts/results_combined/dqn/summary.csv
- run_scripts/results_combined/combined_results.png

## 10. Notes

1. all_run_shell_script.sh is intended to be run with source, which matches the current workflow used in this repository.
2. Small total-timesteps values in run_all.py are suitable for smoke tests only, not for judging final training quality.
3. problem2_ilp/optimal_solution/main.py still depends on an in-script default config filename, so it is not yet a fully parameterized CLI entry point.
4. cleanup_results.py only removes timestamped result directories. It does not remove model zip files or ilp_cache.json.
