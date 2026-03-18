# Problem 2 — ILP Optimal Solution for Service Deployment

## Overview

Problem 2 formulates the service-to-ECU deployment task as an **Integer Linear Programme (ILP)** and solves it to **global optimality** using the PuLP library with the CBC solver.  
The resulting AR value serves as the **theoretical upper bound** for all subsequent RL-based approaches (Problems 3–5, DQN).

---

## Problem Definition

| Symbol               | Meaning                                         |
| -------------------- | ----------------------------------------------- |
| $N$                  | Number of ECUs                                  |
| $M$                  | Number of services ($M \leq N$)                 |
| $e_j$                | VM capacity of ECU $j$                          |
| $n_i$                | VM requirement of service $i$                   |
| $x_{ij} \in \{0,1\}$ | 1 if service $i$ is assigned to ECU $j$, else 0 |

**Objective — maximise total resource utilisation:**

$$\max \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} x_{ij} \cdot \frac{n_i}{e_j}$$

> Note: PuLP requires a linear objective, so the sum is maximised directly.  
> The average AR is computed as `total_utilisation / active_ECUs` after solving.

**Constraints:**

| Constraint           | Mathematical Form                              | Meaning                                    |
| -------------------- | ---------------------------------------------- | ------------------------------------------ |
| Service assignment   | $\sum_j x_{ij} = 1 \;\forall i$                | Every service is placed on exactly one ECU |
| Single-service ECU   | $\sum_i x_{ij} \leq 1 \;\forall j$             | Each ECU hosts at most one service         |
| Capacity feasibility | $x_{ij} = 0$ if $n_i > e_j$                    | Infeasible pairs are pre-blocked           |
| Capacity limit       | $\sum_i x_{ij} \cdot n_i \leq e_j \;\forall j$ | Demand cannot exceed ECU capacity          |

---

## Solution Scenarios

The config files describe four benchmark scenarios:

| Scenario | ECU capacities | Service requirements | Characteristics                          |
| -------- | -------------- | -------------------- | ---------------------------------------- |
| 1        | Uniform        | Uniform              | Symmetric; many valid assignments        |
| 2        | Heterogeneous  | Heterogeneous        | Requires careful matching                |
| 3        | Heterogeneous  | Uniform              | ECU diversity drives utilisation gap     |
| 4        | Uniform        | Heterogeneous        | Service diversity drives utilisation gap |

Each scenario is solved independently; results are compared and plotted.

---

## Output Format

After solving, each scenario produces:

```python
{
    "status":           "Optimal",
    "total_utilization": float,   # sum of ru per active ECU
    "active_ecus":       int,     # number of ECUs that received a service
    "avg_utilization":   float,   # AR = total_utilization / active_ecus
    "allocation": {
        ecu_id: {
            "services":    [svc_id, ...],
            "utilization": float,
            "capacity":    int,
            "demand":      int,
        }
    }
}
```

---

## Comparison with RL Methods

| Dimension               | P2 (ILP)           | P3 (PPO, no constraint)            | P4 (PPO + masking)          | P5 (PPO + Lagrangian)         | DQN              |
| ----------------------- | ------------------ | ---------------------------------- | --------------------------- | ----------------------------- | ---------------- |
| Solution quality        | **Global optimum** | Near-optimal (violations possible) | Near-optimal (0 violations) | Near-optimal (few violations) | Near-optimal     |
| Complexity              | NP-hard            | $O(1)$ inference                   | $O(1)$ inference            | $O(1)$ inference              | $O(1)$ inference |
| Constraint satisfaction | Guaranteed         | ✗                                  | ✓ (hard)                    | ≈ ✓ (soft)                    | ✓ (penalty)      |
| Scalability             | Slow for $N > 100$ | Scales to $N = 10^5$               | Scales                      | Scales                        | Scales           |

---

## Files

| File                        | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `objects.py`                | `ECU` and `SVC` data classes                         |
| `optimal_solution/main.py`  | ILP solver, result printer, summary statistics       |
| `config/generate_config.py` | Generates YAML scenario files                        |
| `config/config_*.yaml`      | Scenario definitions used by all problems            |
| `results/`                  | Cached ILP results (JSON) and per-run output folders |

---

## Usage

```bash
python problem2_ilp/optimal_solution/main.py --config problem2_ilp/config/config_20260305_183222.yaml
```

The script reads all scenarios from the YAML, solves each one with PuLP/CBC, prints per-scenario results, and saves aggregate statistics and plots to `results/`.
