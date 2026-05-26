"""
Shared callbacks for all RL training runs.

EpisodeTrackingCallback:
  - Stops training when episode count reaches target_episodes.
  - Records per-episode cumulative reward (from SB3 Monitor wrapper).
  - Records per-episode info fields (ar, cap/conflict violation bools).
"""

from __future__ import annotations
import time
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeTrackingCallback(BaseCallback):
    """
    Counts completed episodes across all parallel envs.
    Stops training by returning False when target_episodes is reached.
    Collects episode reward and key metrics for plotting.
    """

    def __init__(self, target_episodes: int, progress_every: int | None = None):
        super().__init__()
        self.target_episodes = target_episodes
        # default: every 5% of target
        self.progress_every  = progress_every if progress_every is not None \
            else max(1, target_episodes // 20)

        self.episode_count:   int        = 0
        self.episode_rewards: list[float] = []  # cumulative reward per episode (from Monitor)
        self.episode_ars:     list[float] = []
        self.episode_cap_viol:      list[bool]  = []  # episode had cap violation
        self.episode_conf_viol:     list[bool]  = []  # episode had conflict violation

        self._t_start = 0.0
        self._next_log = self.progress_every

    def _on_training_start(self) -> None:
        self._t_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.episode_count += 1
            self.episode_rewards.append(float(info["episode"]["r"]))
            self.episode_ars.append(float(info.get("ar", 0.0)))
            self.episode_cap_viol.append(bool(info.get("episode_has_cap_violation", False)))
            self.episode_conf_viol.append(bool(info.get("episode_has_conflict_violation", False)))

        if self.episode_count >= self._next_log:
            elapsed = max(time.time() - self._t_start, 1e-6)
            pct = min(100.0, self.episode_count * 100.0 / self.target_episodes)
            eps_per_s = self.episode_count / elapsed
            print(
                f"  [train] ep={self.episode_count:,}/{self.target_episodes:,}"
                f" ({pct:5.1f}%) | eps/s={eps_per_s:,.0f}",
                flush=True,
            )
            self._next_log += self.progress_every

        return self.episode_count < self.target_episodes


class LagrangianUpdateCallback(BaseCallback):
    """
    Dual-ascent λ update for P5 (PPO + Lagrangian).
    Updates lambda every LAMBDA_UPDATE_WINDOW episodes.
    Can be used alongside EpisodeTrackingCallback via CallbackList.
    """

    def __init__(
        self,
        lambda_init: float,
        lambda_lr: float,
        lambda_target: float,
        lambda_max: float,
        update_window: int,
    ):
        super().__init__()
        self.lambda_val    = float(lambda_init)
        self.lambda_lr     = float(lambda_lr)
        self.lambda_target = float(lambda_target)
        self.lambda_max    = float(lambda_max)
        self.update_window = update_window

        self._ep_since_update = 0
        self._window_viols: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            # Use conflict-only violation rate so λ is not polluted by capacity
            # violations (which have their own fixed -2.0 penalty in P5).
            viol_rate = float(info.get("conf_viol_rate_ep",
                                       info.get("viol_rate_ep", 0.0)))
            self._window_viols.append(viol_rate)
            self._ep_since_update += 1

            if self._ep_since_update >= self.update_window:
                avg_viol = sum(self._window_viols) / len(self._window_viols)
                new_lam  = self.lambda_val + self.lambda_lr * (avg_viol - self.lambda_target)
                self.lambda_val = float(max(0.0, min(new_lam, self.lambda_max)))
                self.training_env.env_method("set_lambda", self.lambda_val)
                self._ep_since_update = 0
                self._window_viols.clear()
        return True
