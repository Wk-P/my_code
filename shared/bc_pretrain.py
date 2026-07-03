"""
shared/bc_pretrain.py — ILP behavior-cloning pretraining utilities, reused
across ppo_mask / ppo_lagrangian / ppo_opt / dqn / ddqn pipelines (all lt/eq/gt
scenario variants).

All P4-P6 / DQN / DDQN envs share the same reset() convention: services are
sorted descending by requirement and conflict_sets are remapped accordingly.
Solving the (single-shot) assignment ILP and reordering its allocation to
match that sort gives an expert action for every step of the episode.

Two BC trainers are provided:
  - pretrain_actor_critic(): for SB3 PPO / MaskablePPO (ActorCriticPolicy),
    supervised via -log_prob(expert_action).
  - pretrain_dqn(): for SB3 DQN/DDQN Q-networks, supervised via a DQfD-style
    large-margin classification loss (no log-prob available for a Q-network).
"""

from __future__ import annotations

import numpy as np
import torch

from ilp.objects import ECU, SVC
from shared.ilp_utils import solve_ilp


def ilp_expert_actions(caps, reqs, conflict_sets, sorted_desc=True):
    """Solve the scenario ILP and return the expert action sequence in the
    same service order the target env presents them.

    `sorted_desc` MUST match the target env's reset() convention:
      - True  (default): env sorts services descending by requirement before
        presenting them — this is what every ppo_mask/ppo_lagrangian/ppo_opt
        env does in all scenarios (eq/gt/lt), and what dqn/ddqn do in lt.
      - False: env presents services in their original (unsorted) scenario
        order — this is what dqn/ddqn do in eq/gt (verified by grep: no
        `sort_idx = sorted(...)` in their reset()). Passing the wrong value
        silently misaligns every expert action with the wrong service.

    Returns None if the ILP is infeasible or doesn't place every service."""
    M = len(reqs)
    ecus = [ECU(f"ECU{i}", c) for i, c in enumerate(caps)]
    svcs = [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)]
    res = solve_ilp(ecus, svcs, conflict_sets)
    if res["status"] != "Optimal":
        return None

    svc_to_ecu = {}
    for j, info in res["allocation"].items():
        for i in info["services"]:
            svc_to_ecu[i] = j
    if len(svc_to_ecu) != M:
        return None

    order = sorted(range(M), key=lambda i: -reqs[i]) if sorted_desc else list(range(M))
    return [svc_to_ecu[i] for i in order]


def build_bc_dataset(scenarios, env_cls, use_mask=True, violation_key="total_violations",
                      sorted_desc=True):
    """Replay ILP-optimal actions through `env_cls(ecus, services, scenarios=[sc])`
    and collect (obs, mask, action) triples.

    `sorted_desc` is forwarded to ilp_expert_actions() — see its docstring.
    Passing the wrong value doesn't crash; it silently tanks the "used"
    scenario count (mismatch_skipped) since misaligned expert actions get
    rejected by the mask/violation replay check below. If build_bc_dataset
    logs used=0 or a near-zero fraction, check this first.

    A scenario is kept only if the expert trajectory replays cleanly end-to-end
    (mask check, if `use_mask`) and the episode ends with zero violations
    (checked via `violation_key`, falling back to capacity/conflict counters) —
    this guards against the ILP's constraint encoding drifting from the env's.
    """
    obs_list, mask_list, act_list = [], [], []
    used, skipped_infeasible, skipped_mismatch = 0, 0, 0

    for sc in scenarios:
        caps, reqs, cs = sc[0], sc[1], sc[2] if len(sc) > 2 else []
        actions = ilp_expert_actions(caps, reqs, cs, sorted_desc=sorted_desc)
        if actions is None:
            skipped_infeasible += 1
            continue

        ecus = [ECU(f"ECU{i}", c) for i, c in enumerate(caps)]
        svcs = [SVC(f"SVC{i}", r) for i, r in enumerate(reqs)]
        env = env_cls(ecus, svcs, scenarios=[sc])
        obs, _ = env.reset()

        ep_obs, ep_mask, ep_act = [], [], []
        ok = True
        info = {}
        for a in actions:
            mask = env.action_masks() if use_mask else np.ones(env.N, dtype=bool)
            if use_mask and not mask[a]:
                ok = False
                break
            ep_obs.append(obs)
            ep_mask.append(mask)
            ep_act.append(a)
            obs, _, done, _, info = env.step(a)

        total_viol = info.get(
            violation_key,
            info.get("capacity_violations", info.get("cap_violations", 0))
            + info.get("conflict_violations", 0),
        )

        if ok and total_viol == 0:
            obs_list.extend(ep_obs)
            mask_list.extend(ep_mask)
            act_list.extend(ep_act)
            used += 1
        else:
            skipped_mismatch += 1

    print(f"  [BC data] scenarios used={used}  infeasible_skipped={skipped_infeasible}  "
          f"mismatch_skipped={skipped_mismatch}  transitions={len(act_list)}")
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(mask_list, dtype=bool),
        np.asarray(act_list, dtype=np.int64),
    )


def pretrain_actor_critic(model, obs_arr, act_arr, mask_arr=None,
                           epochs=20, batch_size=256, lr=1e-3):
    """BC for SB3 PPO / MaskablePPO policies (ActorCriticPolicy-based).
    Pass mask_arr for MaskablePPO; leave None for plain PPO (no masking)."""
    policy = model.policy
    device = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs_arr, device=device)
    act_t = torch.as_tensor(act_arr, device=device)
    n = len(act_arr)
    if n == 0:
        print("  [BC] no expert transitions available — skipping pretraining")
        return

    idxs = np.arange(n)
    policy.train()
    for ep in range(epochs):
        np.random.shuffle(idxs)
        total_loss, correct = 0.0, 0
        for start in range(0, n, batch_size):
            b = idxs[start:start + batch_size]
            ob = obs_t[b]
            ac = act_t[b]

            if mask_arr is not None:
                dist = policy.get_distribution(ob, action_masks=mask_arr[b])
            else:
                dist = policy.get_distribution(ob)
            loss = -dist.log_prob(ac).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(b)
            with torch.no_grad():
                pred = dist.distribution.probs.argmax(dim=-1)
                correct += int((pred == ac).sum().item())

        print(f"  [BC] epoch {ep + 1:>2}/{epochs}  loss={total_loss / n:.4f}  acc={correct / n:.4f}")
    policy.eval()


def pretrain_dqn(model, obs_arr, act_arr, epochs=20, batch_size=256, lr=1e-3, margin=0.8):
    """BC warm-start for SB3 DQN/DDQN Q-networks via a DQfD-style large-margin
    classification loss:
        L = max_a[Q(s,a) + margin * 1(a != a_E)] - Q(s, a_E)
    which pushes Q(s, a_E) above every other action by at least `margin`.
    A plain log-prob loss isn't available for Q-networks (no policy distribution).
    """
    q_net = model.q_net
    device = model.device
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs_arr, device=device)
    act_t = torch.as_tensor(act_arr, device=device)
    n = len(act_arr)
    if n == 0:
        print("  [BC-DQN] no expert transitions available — skipping pretraining")
        return

    idxs = np.arange(n)
    q_net.train()
    for ep in range(epochs):
        np.random.shuffle(idxs)
        total_loss, correct = 0.0, 0
        for start in range(0, n, batch_size):
            b = idxs[start:start + batch_size]
            ob = obs_t[b]
            ac = act_t[b].unsqueeze(1)

            q_values = q_net(ob)  # (B, n_actions)
            margin_mask = torch.full_like(q_values, margin)
            margin_mask.scatter_(1, ac, 0.0)
            l_values = q_values + margin_mask
            loss = (l_values.max(dim=1).values - q_values.gather(1, ac).squeeze(1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(b)
            with torch.no_grad():
                pred = q_values.argmax(dim=-1)
                correct += int((pred == ac.squeeze(1)).sum().item())

        print(f"  [BC-DQN] epoch {ep + 1:>2}/{epochs}  loss={total_loss / n:.4f}  acc={correct / n:.4f}")
    q_net.eval()
    # Sync the target network to the freshly pretrained online network so the
    # first training steps don't bootstrap against a random target.
    model.policy.q_net_target.load_state_dict(q_net.state_dict())
