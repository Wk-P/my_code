# Additional Design Notes

## 1. Conflict Condition

Randomly generate 200 conflict sets at initialization. When placing a service on an ECU, check whether the service to be placed and any service already placed on that ECU belong to the same conflict subset. If so, this constitutes a **conflict violation**.

## 2. Updated Placement Constraints

The placement constraints are revised as follows:

- **One ECU may host multiple services** (the one-service-per-ECU limit is removed).

### Two Violation Types

- **Capacity violation**: The total VM requirement of all services placed on an ECU exceeds that ECU's capacity.
  *(Note: Problem 3 does not enforce this constraint — in P3, a reward is given whenever the variable `ar` increases.)*
- **Conflict violation**: If the service being placed and a service already on the same ECU belong to the same conflict subset, the placement is a conflict violation.

## 3. Conflict Set Structure

A conflict set contains K subsets. Each subset contains J services (2 ≤ J ≤ total number of services). Services within the same subset are **mutually exclusive** — they must not be assigned to the same ECU.

Example:
```
{
    {svc1, svc2},
    {svc2, svc3, svc8},
    ...
}
```

## 4. Scenario Set

Generate the same number of scenarios as in `ecu_eq_svc` (200 scenarios). Each scenario must satisfy both of the following conditions:

- The number of ECUs is strictly greater than the number of services: `N > M`
- The total ECU capacity (in VM units) is strictly greater than the total service requirement: `sum(ecu_capacity) > sum(svc_requirement)`
