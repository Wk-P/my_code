# Problem 1: Baseline RL Environment for Service Deployment

## Overview

Problem 1 defines the **base Gym environment** for the service-to-ECU deployment task.  
It is the foundation upon which all subsequent RL approaches (Problem 3–5, DQN) are built.

---

## Problem Definition

| Symbol               | Meaning                                          |
| -------------------- | ------------------------------------------------ |
| $N$                  | Number of ECUs (Electronic Control Units)        |
| $M$                  | Number of services to deploy ($M \leq N$)        |
| $e_j$                | VM capacity of ECU $j$                           |
| $n_i$                | VM requirement of service $i$                    |
| $x_{ij} \in \{0,1\}$ | Decision variable: assign service $i$ to ECU $j$ |

**Objective — maximise Average Resource Utilisation (AR):**

$$\max \; AR = \frac{1}{M} \sum_{j=0}^{N-1} \sum_{i=0}^{M-1} x_{ij} \cdot \frac{n_i}{e_j}$$

**Constraints:**

- Each service is assigned to exactly one ECU.
- Each ECU hosts at most one service.
- The VM requirement of the assigned service must not exceed the ECU capacity.

---

## Environment Design

### State Space

At step $t$, the observation vector has shape $(N + 2)$:

$$s_t = \bigl[\, d_t,\; AR_t,\; r_0^t,\; r_1^t,\; \ldots,\; r_{N-1}^t \,\bigr]$$

| Index          | Content                             | Range   | Description                 |
| -------------- | ----------------------------------- | ------- | --------------------------- |
| 0              | Service demand (normalised)         | $[0,1]$ | $n_t / \max(e_j)$           |
| 1              | Cumulative AR so far                | $[0,1]$ | Running average utilisation |
| $2 \ldots N+1$ | Remaining capacity fraction per ECU | $[0,1]$ | $\text{remaining}_j / e_j$  |

### Action Space

$$a_t \in \{0, 1, \ldots, N-1\}$$

The agent selects which ECU to assign the current service to.

### Reward Function

```
• Infeasible assignment (capacity insufficient): reward = -1.0, episode terminates.
• Valid step but not the last:                   reward =  0.0
• Last valid step (all M services placed):       reward =  final AR
```

This is a **sparse + penalty** design: the agent is discouraged from generating infeasible allocations, and the only positive signal is the final AR at episode completion.

### Episode Structure

```
reset()
  └── step 0 : assign service_0 → ECU[a_0]
       step 1 : assign service_1 → ECU[a_1]
       ...
       step M-1 : assign service_{M-1} → ECU[a_{M-1}]  → reward = AR
```

If the agent picks an ECU with insufficient remaining capacity, the episode terminates immediately with `reward = -1`.

---

## Key Classes

| Class        | File     | Description                                       |
| ------------ | -------- | ------------------------------------------------- |
| `my_ecu`     | `env.py` | ECU with a fixed VM capacity                      |
| `my_service` | `env.py` | Service with a VM requirement                     |
| `my_env`     | `env.py` | Gymnasium environment wrapping the assignment MDP |

---

## Design Notes

- **Termination on infeasible action** acts as an implicit hard constraint.  
  The agent learns to avoid capacity overflows without explicit masking.
- **Remaining capacity tracking** in the observation allows the agent to make greedy-like decisions (assign large services to large ECUs).
- The observation space bounds are `[0.0, 1.0]` for all dimensions.
- This environment is intended as a **teaching baseline**; later problems relax or modify the constraint-handling strategy.
