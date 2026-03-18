from os import read
from pathlib import Path
import sys, time, csv
import numpy as np
import json 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).parent
ROOT_PATH = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_PATH))


# read csv files from paths

PATH_LIST = [
    HERE / "problem3_ppo" / "summary.csv",
    HERE / "problem4_ppo_mask" / "summary.csv",
    HERE / "problem5_ppo_lagrangian" / "summary.csv",
    HERE / "problem6_ppo_opt" / "summary.csv",
    HERE / "dqn" / "summary.csv",
]

ILP_DATA_PATH = HERE / "ilp_data.csv"

head_text_width = 10
print(f"{'HERE:':<{head_text_width}} {HERE}")
print(f"{'ROOT_PATH:':<{head_text_width}} {ROOT_PATH}")



"""This script combines results from all problems (3-6 PPO variants + DQN) and ILP into a single CSV file for easier comparison and plotting."""

def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def read_csv(path: Path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)
    

def plot_combined_results(data_list, ilp_data, output_path: Path):
    # ax are [AR, total_violations, capacity_violations, single_service_violations]
    # ax1: p3-6 AR, ax2: p3-6 total violations, ax3: p3-6 capacity violations, ax4: p3-6 single service violations
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)  
    problem_names = ["P3 PPO", "P4 PPO Mask", "P5 PPO Lagrangian", "P6 PPO Opt", "DQN"]
    colors = ["blue", "orange", "green", "red", "purple"]
    for data, name, color in zip(data_list, problem_names, colors):
        ars = [float(row["AR"]) for row in data]
        total_viols = [int(row["total_violations"]) for row in data]
        cap_viols = [int(row["capacity_violations"]) for row in data]
        single_service_viols = [int(row["single_service_violations"]) for row in data]

        ax1.bar(name, np.mean(ars), color=color)
        ax2.bar(name, np.mean(total_viols), color=color)
        ax3.bar(name, np.mean(cap_viols), color=color)
        ax4.bar(name, np.mean(single_service_viols), color=color)



    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main():
    combined_data = []
    for path in PATH_LIST:
        data = read_csv(path)
        combined_data.append(data)
    
    ilp_data = read_json(ILP_DATA_PATH)
    
    # draw combined results
    plot_combined_results(combined_data, ilp_data, output_path=HERE / "combined_results.png")

    
    
if __name__ == "__main__":
    main()