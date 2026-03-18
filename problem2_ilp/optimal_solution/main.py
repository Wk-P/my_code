import pulp
import random
import json
from pathlib import Path


# Create a linear programming problem:
""" 
Problem Description:
There are {M} services and {N} ECUs, every ECU has {e} vm capacity and every service has {x} vm requirement. 
The goal is to maxmize the average resource utilization of all ECUs while ensuring {M} services are planned to deploy on the {N} ECUs.
single ECU's resource utilization  = (vm requirement of one service) / (vm capacity of the used ECU)
average resource utilization of all ECUs = (sum of single EUC's resource utilization) / (actived number of ECUs)
and we have the constaint that the vm requirement of the services assigned to an ECU cannot exceed the vm capacity of that ECU.

IMPORTANT: Each ECU can only be assigned ONE service (not multiple services)

Situation 1:
{e} are same
{x} are same

Situation 2:
{e} are different
{x} are different

Situation 3:
{e} are different
{x} are same

Situation 4:
{e} are same
{x} are different
"""

def read_config(config_path):
    """Read configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def solve_service_deployment(M, N, e_list, n_list, seed=None):
    """
    Solve the service deployment optimization problem
    
    Parameters:
    - M: number of services
    - N: number of ECUs
    - e_list: list of ECU vm capacity (length N)
    - n_list: list of service vm requirement (length M)
    - seed: random seed (optional)
    
    Returns:
    - Dictionary containing optimization results
    """
    
    if seed is not None:
        random.seed(seed)
    
    # check {M}, {N}
    if M > N:
        raise ValueError("Number of services (M) cannot exceed number of ECUs (N) since each ECU can only host one service.")
    
    # Create a linear programming problem
    problem = pulp.LpProblem("Maximize_Average_Resource_Utilization", pulp.LpMaximize)
    
    # Create decision variables
    # x[i][j] = 1 if service i is assigned to ECU j, 0 otherwise
    x = pulp.LpVariable.dicts("x", (range(M), range(N)), cat='Binary')
    
    # Objective function: Maximize total resource utilization (linear objective)
    # NOTE: Do not divide by a decision-variable expression in PuLP.
    problem += pulp.lpSum((x[i][j] * n_list[i] / e_list[j]) for i in range(M) for j in range(N)), "Total_Resource_Utilization"
    
    # Constraints
    # Each service must be assigned to exactly one ECU
    for i in range(M):
        problem += pulp.lpSum(x[i][j] for j in range(N)) == 1, f"Service_{i}_Assignment"
    
    # Each ECU can only be assigned at most ONE service
    for j in range(N):
        problem += pulp.lpSum(x[i][j] for i in range(M)) <= 1, f"ECU_{j}_Single_Service"
    
    # Pairwise capacity feasibility:
    # For every (service i, ECU j), if service demand exceeds ECU capacity, assignment is forbidden.
    for i in range(M):
        for j in range(N):
            if n_list[i] > e_list[j]:
                problem += x[i][j] == 0, f"Infeasible_Assignment_s{i}_e{j}"
    
    # The vm requirement of the service assigned to an ECU cannot exceed its capacity
    for j in range(N):
        problem += pulp.lpSum(x[i][j] * n_list[i] for i in range(M)) <= e_list[j], f"ECU_{j}_Capacity"
    
    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Collect results
    result = {
        'status': pulp.LpStatus[problem.status],
        'total_utilization': pulp.value(problem.objective),
        'active_ecus': 0,
        'avg_utilization': 0.0,
        'allocation': {}
    }
    
    for j in range(N):
        services = [i for i in range(M) if pulp.value(x[i][j]) == 1]
        if services:
            utilization = sum(n_list[i] for i in services) / e_list[j]
            result['allocation'][j] = {
                'services': services,
                'utilization': utilization,
                'capacity': e_list[j],
                'demand': sum(n_list[i] for i in services)
            }

    # Calculate active ECU count and average utilization after solving
    result['active_ecus'] = len(result['allocation'])
    if result['active_ecus'] > 0:
        result['avg_utilization'] = result['total_utilization'] / result['active_ecus']
    
    return result


def print_result(result, case_name):
    """Print the results"""
    print(f"\n{'='*60}")
    print(f"Case: {case_name}")
    print(f"{'='*60}")
    print(f"Status: {result['status']}")
    print(f"Average Resource Utilization: {result['avg_utilization']:.2%}")
    
    for ecu_id, info in sorted(result['allocation'].items()):
        print(f"ECU {ecu_id}: services{info['services']} | utilization{info['utilization']:.2%} ({info['demand']}/{info['capacity']})")


def _generate_summary_statistics(results, output_dir):
    """Generate aggregate summary statistics and visualization across all scenarios"""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from scipy import stats

    if not results:
        print("No results to summarize")
        return

    # Extract metrics from all scenarios
    avg_utilizations = [r['avg_utilization'] * 100 for r in results]
    total_utilizations = [r['total_utilization'] * 100 for r in results]
    statuses = [r['status'] for r in results]
    optimal_count = sum(1 for s in statuses if 'Optimal' in s)

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'font.size': 12})

    # Top row: 2 charts; bottom row: table spanning full width
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.5
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.5
    plt.rcParams['figure.constrained_layout.hspace'] = 0.08
    plt.rcParams['figure.constrained_layout.wspace'] = 0.08
    fig = plt.figure(figsize=(19, 11), layout="constrained", dpi=300)
    fig.suptitle('Service Deployment Optimization - Summary Statistics',
                 fontsize=28, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[2.8, 1])

    # ── Chart 1: Histogram + KDE ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(avg_utilizations, bins=20,
                                color='#3498db', alpha=0.7,
                                edgecolor='black', linewidth=1.0)
    import matplotlib.patches as mpatches
    patch_list: list[mpatches.Rectangle] = list(patches)  # type: ignore[arg-type]
    for i, patch in enumerate(patch_list):
        mid = (bins[i] + bins[i + 1]) / 2
        if mid >= 60:
            patch.set_facecolor('#2ecc71')
        elif mid >= 50:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#e74c3c')

    density = stats.gaussian_kde(avg_utilizations)
    x_range = np.linspace(min(avg_utilizations), max(avg_utilizations), 300)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_range, density(x_range), 'b-', linewidth=2.5, alpha=0.85)
    ax1_twin.set_ylabel('Density', fontsize=13)
    ax1_twin.tick_params(labelsize=12)
    ax1_twin.grid(False)

    mean_util   = float(np.mean(avg_utilizations))
    median_util = float(np.median(avg_utilizations))
    ax1.axvline(mean_util,   color='red',   linestyle='--', linewidth=2,
                label=f'Mean: {mean_util:.1f}%')
    ax1.axvline(median_util, color='green', linestyle=':',  linewidth=2,
                label=f'Median: {median_util:.1f}%')

    ax1.set_xlabel('Average Utilization (%)', fontsize=13)
    ax1.set_ylabel('Frequency', fontsize=13)
    ax1.set_title('Distribution of Average Resource Utilization',
                  fontsize=16, fontweight='bold', pad=10)
    ax1.tick_params(labelsize=12)
    ax1.legend(loc='upper left', fontsize=12)

    # ── Chart 2: Box plot + jitter ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(avg_utilizations,
                     positions=[1], widths=0.5,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='yellow',
                                   markeredgecolor='black', markersize=8))

    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][0].set_linewidth(1.5)
    for w in bp['whiskers']:
        w.set(linewidth=1.5, color='#34495e')
    bp['medians'][0].set(linewidth=2.5, color='red')

    jitter = np.random.normal(1, 0.05, size=len(avg_utilizations))
    ax2.scatter(jitter, avg_utilizations, alpha=0.35, s=30,
                color='#2ecc71', zorder=3, label='Each scenario')

    ax2.set_xticks([1])
    ax2.set_xticklabels(['Avg Util per ECU (%)'], fontsize=13)
    ax2.set_ylabel('Utilization (%)', fontsize=13)
    ax2.set_title('Avg Utilization Distribution (Box + Scatter)',
                  fontsize=16, fontweight='bold', pad=10)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.legend(fontsize=12)

    q1, q3 = np.percentile(avg_utilizations, [25, 75])
    textstr = (f"Min  = {min(avg_utilizations):.1f}%\n"
               f"Q1   = {q1:.1f}%\n"
               f"Med  = {median_util:.1f}%\n"
               f"Q3   = {q3:.1f}%\n"
               f"Max  = {max(avg_utilizations):.1f}%\n"
               f"Std  = {np.std(avg_utilizations):.1f}%")
    mean_y = float(np.mean(avg_utilizations))
    ax2.text(1.42, mean_y, textstr,
             fontsize=12, va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                       edgecolor='#bdc3c7', alpha=0.9))

    # ── Table: spanning full bottom row ─────────────────────────────────────
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')

    stats_data = [
        ['Metric',         'Min',   'Max',   'Mean',  'Std',  'Scenarios', 'Optimal Rate'],
        ['Avg Util (%)',
         f"{min(avg_utilizations):.1f}",
         f"{max(avg_utilizations):.1f}",
         f"{np.mean(avg_utilizations):.1f}",
         f"{np.std(avg_utilizations):.1f}",
         str(len(results)),
         f"{optimal_count/len(results)*100:.0f}%"],
        ['Total Util (%)',
         f"{min(total_utilizations):.1f}",
         f"{max(total_utilizations):.1f}",
         f"{np.mean(total_utilizations):.1f}",
         f"{np.std(total_utilizations):.1f}",
         '', ''],
    ]

    col_widths = [0.16, 0.10, 0.10, 0.10, 0.10, 0.13, 0.13]
    table = ax_table.table(cellText=stats_data, cellLoc='center',
                           loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.4)

    # Header row
    for j in range(len(stats_data[0])):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white', fontsize=14)
    # Data rows
    for i in range(1, len(stats_data)):
        bg = '#ffffff' if i % 2 == 1 else '#ecf0f1'
        for j in range(len(stats_data[i])):
            cell = table[(i, j)]
            cell.set_facecolor(bg)
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=13)

    ax_table.set_title('Summary Statistics', fontsize=16, fontweight='bold', pad=10)

    # Save
    summary_path = output_dir / 'summary_statistics.png'
    plt.savefig(summary_path, dpi=150)
    print(f"\n✓ Summary statistics saved to {summary_path.name}")
    plt.close()

    # Console summary
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total Scenarios   : {len(results)}")
    print(f"Optimal Solutions : {optimal_count}/{len(results)} ({optimal_count/len(results)*100:.0f}%)")
    print(f"\nAverage Utilization:")
    print(f"  Min    : {min(avg_utilizations):.2f}%")
    print(f"  Max    : {max(avg_utilizations):.2f}%")
    print(f"  Mean   : {np.mean(avg_utilizations):.2f}%")
    print(f"  Std Dev: {np.std(avg_utilizations):.2f}%")
    print(f"{'='*60}\n")



def main(**kwargs):
    import datetime
    
    # read config from YAML file
    config_path = kwargs.get('config_path', None)
    
    # If no path provided, find the latest config file
    if config_path is None:
        config_dir = Path(__file__).parent.parent / 'config'
        config_files = sorted(config_dir.glob('config_*.yaml'), reverse=True)
        if config_files:
            config_path = config_files[0]
            print(f"No config path specified, using latest: {config_path.name}")
        else:
            # Fallback to default config.yaml
            config_path = config_dir / 'config.yaml'
    else:
        config_path = Path(config_path)
    
    config_data = read_config(config_path)
    
    # Create output directory with timestamp
    output_base_dir = Path(__file__).parent.parent / 'results'
    output_base_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f'result_{timestamp}'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}\n")

    # Handle both single config and multiple scenarios
    # If config_data is a dict with 'scenarios' key, it contains multiple environments
    if isinstance(config_data, dict) and 'scenarios' in config_data:
        scenarios = config_data['scenarios']
    else:
        # Single scenario (backward compatibility)
        scenarios = [config_data]

    # ── Shared ILP cache (used by p3/p4/dqn/p5 to avoid recomputation) ──────
    shared_cache_path = output_base_dir / "ilp_cache.json"
    cache_key = f"{config_path.name}__n{len(scenarios)}"

    # Load partial or full cache if available
    results: list = []
    if shared_cache_path.exists():
        try:
            with open(shared_cache_path) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}
        if cache.get("key") == cache_key:
            results = cache.get("results", [])
            already = len(results)
            if already == len(scenarios):
                print(f"[cache] All {already} ILP results loaded from {shared_cache_path.name}")
                _generate_summary_statistics(results, output_dir)
                return results
            print(f"[cache] Resuming from scenario {already + 1} "
                  f"({already}/{len(scenarios)} already cached)")

    # Process each scenario (skip already-cached ones)
    start_idx = len(results)
    for idx, scenario in enumerate(scenarios[start_idx:], start=start_idx):
        scenario_name = scenario.get('name', f'Scenario {idx+1}')
        
        print(f"\n{'='*60}")
        print(f"Processing: {scenario_name}")
        print(f"{'='*60}")

        # extract M, N, e_list, n_list from scenario
        M = len(scenario['SVCs'])
        N = len(scenario['ECUs'])
        e_list = [ecu['capacity'] for ecu in scenario['ECUs']]
        n_list = [svc['requirement'] for svc in scenario['SVCs']]

        print(f"Number of Services (M): {M}")
        print(f"Number of ECUs (N): {N}")
        print(f"ECU Capacities: {e_list}")
        print(f"Service Requirements: {n_list}")

        # Solve the deployment problem
        result = solve_service_deployment(M, N, e_list, n_list)
        result['scenario_name'] = scenario_name
        result['M'] = M
        result['N'] = N
        results.append(result)
        print_result(result, scenario_name)

        # Save incrementally so interruptions don't lose progress (atomic write)
        tmp = shared_cache_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"key": cache_key, "results": results}, f)
        tmp.replace(shared_cache_path)

    print(f"\n{'='*60}")
    print(f"All scenarios processed. Total: {len(results)}")
    print(f"Shared ILP cache saved to: {shared_cache_path}")
    print(f"Output saved to: {output_dir.resolve()}")
    print(f"{'='*60}")
    
    # Generate summary statistics and visualizations
    _generate_summary_statistics(results, output_dir)
    
    return results

if __name__ == "__main__":
    import sys
    
    # Check if config filename is provided as command line argument
    config_filename = "config_20260305_183222"  # Default config filename (can be overridden by user input)
    
    # Build full path
    if not config_filename.endswith('.yaml'):
        config_filename += '.yaml'
    
    config_path = Path(__file__).parent.parent / 'config' / config_filename
    
    # Verify file exists
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"\nUsing config file: {config_path.name}\n")
    main(config_path=str(config_path))