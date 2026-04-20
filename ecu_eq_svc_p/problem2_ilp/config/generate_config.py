import sys
from pathlib import Path

# Add project root to sys.path so we can import problem2 modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import yaml
from problem2_ilp.objects import ECU, SVC
import datetime
import random

N_ECUS = 10  # ECU count == SVC count (ecu_eq_svc)
N_SVCS = 10


def generate_env(scenario_id):
    """Generate a single environment configuration"""
    ecu_capacity = random.sample(range(50, 200, 5), N_ECUS)
    svc_requirement = random.sample(range(10, 100, 5), N_SVCS)

    ecu_list = [ECU(f"ECU{i}", capacity) for i, capacity in enumerate(ecu_capacity)]
    svc_list = [SVC(f"SVC{i}", requirement) for i, requirement in enumerate(svc_requirement)]

    K = 10
    conflict_sets = []
    for _ in range(K):
        size = random.randint(2, N_SVCS)
        conflict_sets.append(random.sample(range(N_SVCS), size))

    config = {
        "name": f"Scenario {scenario_id}",
        "generated_by": "generate_config.py",
        "generated_on": datetime.datetime.now().isoformat(),
        "ECUs": [ecu.__dict__() for ecu in ecu_list],
        "SVCs": [svc.__dict__() for svc in svc_list],
        "conflict_sets": conflict_sets,
    }

    print(f"Generated {config['name']}: {len(ecu_list)} ECUs, {len(svc_list)} SVCs")
    return config


def write_config(scenarios, filename=None):
    """Write multiple scenarios to YAML file"""
    config = {
        "generated_by": "generate_config.py",
        "generated_on": datetime.datetime.now().isoformat(),
        "scenarios": scenarios
    }
    
    # Generate filename with timestamp to avoid overwriting
    path_config_directory = Path(__file__).parent
    new_file_path = path_config_directory / f'config_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'

    with open(new_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
    print(f"\nConfiguration file saved to: {new_file_path.resolve()}")
    print(f"Total scenarios: {len(scenarios)}")
    

if __name__ == "__main__":
    # Generate n different environment scenarios
    n = 200  # Number of scenarios to generate
    scenarios = []
    
    print(f"Generating {n} scenarios...")
    for i in range(n):
        env = generate_env(i + 1)
        scenarios.append(env)
    
    write_config(scenarios, None)