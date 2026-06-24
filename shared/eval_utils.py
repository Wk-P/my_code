"""
Shared evaluation logic used by run_experiment.py.

evaluate_model() runs one episode per test scenario and returns a dict with:
  - ar:                    float (average resource utilisation)
  - success:               bool  (all M services placed AND no violation of any kind)
  - episode_has_cap_violation:      bool
  - episode_has_conflict_violation: bool
  - services_placed:       int
  - M:                     int (total services)

success definition:
  services_placed == M  AND  NOT episode_has_cap_violation  AND  NOT episode_has_conflict_violation
"""

from __future__ import annotations
import numpy as np
from typing import Callable


def evaluate_model(
    policy_fn: Callable,           # obs -> action   (or obs, mask -> action for P4)
    test_scenarios: list,
    env_cls,                        # e.g. P3Env, P4Env, …
    env_kwargs: dict | None = None, # extra kwargs forwarded to env_cls.__init__
    needs_mask: bool = False,       # True for MaskablePPO / P4
) -> list[dict]:
    """
    Returns one result dict per test scenario.
    """
    env_kwargs = env_kwargs or {}
    from ilp.objects import ECU, SVC

    results = []
    for scenario in test_scenarios:
        caps, reqs, cs = scenario
        ecus = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
        svcs = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
        env = env_cls(ecus, svcs, scenarios=[scenario], **env_kwargs)
        obs, _ = env.reset()
        done = False
        info: dict = {}

        while not done:
            if needs_mask:
                mask = env.action_masks()
                if not np.any(mask):
                    break
                action = policy_fn(obs, mask)
            else:
                action = policy_fn(obs)
            obs, _, done, _, info = env.step(int(action))

        has_cap  = bool(info.get("episode_has_cap_violation", False))
        has_conf = bool(info.get("episode_has_conflict_violation", False))
        placed   = int(info.get("services_placed", 0))
        M        = env.M

        results.append({
            "ar":                             float(info.get("ar", 0.0)),
            "success":                        (placed == M) and (not has_cap) and (not has_conf),
            "episode_has_cap_violation":      has_cap,
            "episode_has_conflict_violation": has_conf,
            "services_placed":                placed,
            "M":                              M,
        })

    return results


def aggregate_eval(per_scenario_results: list[dict]) -> dict:
    """Aggregate a list of per-scenario result dicts into summary statistics."""
    n = len(per_scenario_results)
    if n == 0:
        return {}
    ar_vals     = [r["ar"]      for r in per_scenario_results]
    success     = [r["success"] for r in per_scenario_results]
    cap_viol    = [r["episode_has_cap_violation"]      for r in per_scenario_results]
    conf_viol   = [r["episode_has_conflict_violation"] for r in per_scenario_results]
    placed      = [r["services_placed"] for r in per_scenario_results]
    M           = per_scenario_results[0]["M"]
    return {
        "ar_mean":           float(np.mean(ar_vals)),
        "ar_std":            float(np.std(ar_vals)),
        "success_rate":      float(np.mean(success)),
        "failure_rate":      float(1.0 - np.mean(success)),
        "cap_viol_rate":     float(np.mean(cap_viol)),
        "conf_viol_rate":    float(np.mean(conf_viol)),
        "placed_mean":       float(np.mean(placed)),
        "M":                 M,
        "n_scenarios":       n,
    }
