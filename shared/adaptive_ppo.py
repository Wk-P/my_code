"""
Two experiment knobs shared by problem4 (MaskablePPO) and problem5 (PPO):

1. entropy_at(step, total_steps, init, final) — linear entropy-coefficient
   schedule (start exploratory, anneal down), replacing the static
   PPO_ENT_COEF constant. Motivated by Hamalainen et al. 2020 (PPO-CMA):
   a fixed/too-small entropy coefficient lets PPO's exploration collapse
   prematurely and get stuck in local optima.

2. PrunedPPO / PrunedMaskablePPO — PPO/MaskablePPO subclasses that
   down-weight negative-advantage samples in the policy loss (CMA-ES-style
   selection, where only the better half of sampled actions drives the
   mean/covariance update). Vanilla PPO instead lets every negative-advantage
   sample push the policy away from that action with full weight, which can
   destabilise updates in this combinatorial action space. Enabled via
   adv_prune_weight < 1.0 (1.0 = vanilla PPO, no pruning).

Both classes copy stable_baselines3==2.7.1 / sb3_contrib's `train()` verbatim
except for the one weighting line, so behaviour matches upstream PPO exactly
when adv_prune_weight=1.0.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from sb3_contrib import MaskablePPO


def _linear_anneal(step: int, total_steps: int, init: float, final: float) -> float:
    """Linear anneal from `init` (at step=0) to `final` (at step>=total_steps)."""
    if total_steps <= 0:
        return final
    progress = min(1.0, max(0.0, step / total_steps))
    return init + (final - init) * progress


def entropy_at(step: int, total_steps: int, init: float, final: float) -> float:
    """Linear anneal from `init` (at step=0) to `final` (at step>=total_steps)."""
    return _linear_anneal(step, total_steps, init, final)


def beta_at(step: int, total_steps: int, init: float, final: float) -> float:
    """Linear anneal for the bottleneck_risk potential-shaping weight
    (P4Env's `bottleneck_shaping_weight`), same schedule shape as
    entropy_at() but decaying (init high, final low/zero) instead of
    annealing down like entropy.

    Why decay it at all, given potential-based shaping is provably
    policy-invariant at ANY fixed beta>=0 (Ng, Harada & Russell 1999): that
    guarantee is an asymptotic/exact-optimization result. Mid-training, with
    a finite sample budget, a large fixed beta adds real variance to the
    per-step reward the value function has to fit, and the shaping/task-
    reward now have to be correctly disentangled every gradient step instead
    of just once at convergence -- a plain constant beta=2.0 measured worse
    on both success_rate and AR than beta=0.0 in an ablation (500k steps),
    consistent with the shaping term acting as an under-fit distraction
    rather than a helpful prior this early. Annealing beta -> 0 over
    training gets the early-training guidance (when the policy is closest
    to uniform-random and most likely to wander into dead ends) while
    letting the back half of training converge on the pure M*ar/-M signal,
    so the final policy isn't fit against a residual shaping term at all.
    """
    return _linear_anneal(step, total_steps, init, final)


def _adv_weights(advantages: th.Tensor, prune_weight: float) -> th.Tensor:
    """1.0 for non-negative-advantage samples, `prune_weight` for negative ones."""
    if prune_weight >= 1.0:
        return th.ones_like(advantages)
    return th.where(advantages > 0, th.ones_like(advantages), th.full_like(advantages, prune_weight))


class PrunedPPO(PPO):
    """PPO (used by problem5 ppo_lagrangian) with negative-advantage down-weighting."""

    def __init__(self, *args, adv_prune_weight: float = 1.0, **kwargs):
        self.adv_prune_weight = adv_prune_weight
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                weights = _adv_weights(advantages, self.adv_prune_weight)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                per_sample_loss = -th.min(policy_loss_1, policy_loss_2)
                policy_loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class PrunedMaskablePPO(MaskablePPO):
    """MaskablePPO (used by problem4 ppo_mask) with negative-advantage down-weighting."""

    def __init__(self, *args, adv_prune_weight: float = 1.0, **kwargs):
        self.adv_prune_weight = adv_prune_weight
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, action_masks=rollout_data.action_masks,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                weights = _adv_weights(advantages, self.adv_prune_weight)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                per_sample_loss = -th.min(policy_loss_1, policy_loss_2)
                policy_loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
