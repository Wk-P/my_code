# Best-Fit Patch Algorithm for Duplicate ECU Assignment

## Overview

When the RL agent selects an ECU (`ecu_n`) that is already occupied by a service (`s1`),
instead of simply recording a violation and overwriting, this algorithm **reschedules** both
services across `ecu_n` and a chosen free ECU (`ecu_m`) to minimise constraint violations
and maximise resource utilisation.

---

## Required State (additions to the environment)

| Variable          | Type                | Description                                                      |
| ----------------- | ------------------- | ---------------------------------------------------------------- |
| `ecu_service_idx` | `int[N]`, init `-1` | Index of the service currently running on each ECU (`-1` = free) |

This must be initialised to `-1` in `reset()` and updated on every valid assignment.

---

## Trigger Condition

```
ecu_assigned[ecu_n] == True
```

- `s1` = `services[ecu_service_idx[ecu_n]]` — service already on `ecu_n`
- `s2` = `svc` — newly arriving service (current step)

---

## Step 1 — Determine `s_stay` and `s_move`

Compare `s1.req` and `s2.req`:

**Case A: `s2.req > s1.req`** (new service is larger)

- `s_stay = s2` (larger, stays on `ecu_n`)
- `s_move = s1` (smaller, will be relocated)
- Extra check: if `initial_vms[ecu_n] < s2.req` → `capacity_violations += 1`

**Case B: `s2.req <= s1.req`** (new service is smaller or equal)

- `s_stay = s1` (larger, stays on `ecu_n`)
- `s_move = s2` (smaller, will be relocated)
- No capacity check needed (`s1` was already legally placed on `ecu_n`)

---

## Step 2 — Find Best-Fit `ecu_m` (Normal Path)

Search all **free** ECUs (`ecu_assigned[j] == False`, `j != ecu_n`):

```
candidates = [j for j in range(N) if not ecu_assigned[j] and initial_vms[j] >= s_move.req]
ecu_m = argmin over candidates of (remaining_vms[j] - s_move.req)   # tightest fit
```

- **Found** → proceed to Step 3.
- **Not found** → proceed to Step 5 (Fallback).

---

## Step 3 — Execute Relocation (Normal Path)

```
remaining_vms[ecu_n]  = initial_vms[ecu_n] - s_stay.req
remaining_vms[ecu_m] -= s_move.req

ecu_service_idx[ecu_n] = index of s_stay
ecu_service_idx[ecu_m] = index of s_move
ecu_assigned[ecu_m]    = True
```

---

## Step 4 — Correct the AR (Normal Path)

`s_move`'s RU contribution has changed from `cap_n` to `cap_m`.  
Apply a delta correction to the running AR (without re-scanning all steps):

$$AR_{\text{new}} = AR_{\text{old}} + \frac{s\_move.req}{self.\_step} \cdot \left(\frac{1}{cap_m} - \frac{1}{cap_n}\right)$$

> `self._step` here is the number of services **already committed** before the current step,
> i.e. the denominator used when `s_move` was originally counted into `AR`.

---

## Step 5 — Fallback (No Capacity-Sufficient Free ECU Found)

Enumerate all free ECUs as candidates for `ecu_m`.  
For each candidate `ecu_m`, evaluate **both arrangements**:

| Arrangement | ECU `ecu_n` runs | ECU `ecu_m` runs |
| ----------- | ---------------- | ---------------- |
| X           | s1               | s2               |
| Y           | s2               | s1               |

For each `(ecu_m, arrangement)` pair compute:

**Violation count** (0, 1, or 2):

```
viol = 0
if s_on_ecu_n.req > initial_vms[ecu_n]:  viol += 1
if s_on_ecu_m.req > initial_vms[ecu_m]:  viol += 1
```

**Score** (higher is better):
$$\text{score} = \frac{s1.req + s2.req}{cap_n + cap_m}$$

**Selection rule** (two-level sort):

1. Minimise violation count — **combinations with 2 violations are never selected**
2. Among equal violation counts, maximise score

Execute the selected arrangement, add the actual violation count to `capacity_violations`.  
`single_service_violations` is **not** incremented (the two services end up on separate ECUs).

---

## Summary Table

| Situation                                  | Path     | `capacity_violations` | `single_service_violations` |
| ------------------------------------------ | -------- | --------------------- | --------------------------- |
| Normal, no cap issue on `ecu_n` (Case B)   | Normal   | +0                    | +0                          |
| Normal, cap issue on `ecu_n` (Case A)      | Normal   | +1                    | +0                          |
| Fallback, solution found with 0 violations | Fallback | +0                    | +0                          |
| Fallback, solution found with 1 violation  | Fallback | +1                    | +0                          |
| Two-violation combinations                 | —        | never selected        | —                           |
