"""
Callbacks for SAC training on the macro environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMacroMetricsCallback(BaseCallback):
    """
    Track episode-level macro statistics and write them to the logger/JSONL.
    """

    def __init__(self, metrics_path: str | None = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self._episode_state_buffers: list[dict[str, list[float]]] = []
        self._episode_reward_buffers: list[list[float]] = []

    def _on_training_start(self) -> None:
        num_envs = self.training_env.num_envs
        self._episode_state_buffers = [self._empty_state_buffer() for _ in range(num_envs)]
        self._episode_reward_buffers = [[] for _ in range(num_envs)]
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for env_idx, info in enumerate(infos):
            state = info.get("state")
            if state is not None:
                self._append_state(env_idx, state)
            if env_idx < len(rewards):
                self._episode_reward_buffers[env_idx].append(float(rewards[env_idx]))

            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            if done:
                self._finalize_episode(env_idx, info)

        return True

    def _finalize_episode(self, env_idx: int, info: dict) -> None:
        state_buffer = self._episode_state_buffers[env_idx]
        rewards = self._episode_reward_buffers[env_idx]
        episode_info = info.get("episode", {})

        metrics = {
            "mean_inflation": self._safe_mean(state_buffer["pi"]),
            "std_inflation": self._safe_std(state_buffer["pi"]),
            "mean_unemployment": self._safe_mean(state_buffer["u"]),
            "std_unemployment": self._safe_std(state_buffer["u"]),
            "mean_growth": self._safe_mean(state_buffer["g"]),
            "mean_debt": self._safe_mean(state_buffer["d"]),
            "max_debt": max(state_buffer["d"]) if state_buffer["d"] else 0.0,
            "return": float(episode_info.get("r", sum(rewards))),
            "length": int(episode_info.get("l", len(rewards))),
            "timesteps": int(self.num_timesteps),
        }

        self.logger.record("episode/mean_inflation", metrics["mean_inflation"])
        self.logger.record("episode/std_inflation", metrics["std_inflation"])
        self.logger.record("episode/mean_unemployment", metrics["mean_unemployment"])
        self.logger.record("episode/std_unemployment", metrics["std_unemployment"])
        self.logger.record("episode/mean_growth", metrics["mean_growth"])
        self.logger.record("episode/mean_debt", metrics["mean_debt"])
        self.logger.record("episode/max_debt", metrics["max_debt"])
        self.logger.record("episode/return", metrics["return"])
        self.logger.record("episode/length", metrics["length"])

        if self.metrics_path is not None:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")

        self._episode_state_buffers[env_idx] = self._empty_state_buffer()
        self._episode_reward_buffers[env_idx] = []

    @staticmethod
    def _empty_state_buffer() -> dict[str, list[float]]:
        return {
            "pi": [],
            "u": [],
            "g": [],
            "d": [],
        }

    def _append_state(self, env_idx: int, state: dict) -> None:
        buffer = self._episode_state_buffers[env_idx]
        buffer["pi"].append(float(state["pi"]))
        buffer["u"].append(float(state["u"]))
        buffer["g"].append(float(state["g"]))
        buffer["d"].append(float(state["d"]))

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _safe_std(values: list[float]) -> float:
        return float(np.std(values)) if values else 0.0
