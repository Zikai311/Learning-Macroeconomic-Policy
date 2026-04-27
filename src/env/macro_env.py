"""
Gymnasium wrapper around the macroeconomic simulator.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.models.economy import Economy
from src.utils.config import EconomyConfig, RewardConfig


OBSERVATION_KEYS = ("pi", "u", "g", "r", "d", "E_pi", "tau", "G")


class MacroEnv(gym.Env):
    """
    Gymnasium-compatible macroeconomic environment.

    Observation vector:
        [pi, u, g, r, d, E_pi, tau, G]

    Action vector:
        [delta_r, delta_G, delta_tau]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[EconomyConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EconomyConfig()
        self.reward_config = reward_config or RewardConfig()
        self.economy = Economy(config=self.cfg, reward_config=self.reward_config)

        self.observation_space = spaces.Box(
            low=np.array([
                self.cfg.pi_bounds[0],
                self.cfg.u_bounds[0],
                self.cfg.g_bounds[0],
                self.cfg.r_bounds[0],
                self.cfg.d_bounds[0],
                self.cfg.E_pi_bounds[0],
                self.cfg.tau_bounds[0],
                self.cfg.G_bounds[0],
            ], dtype=np.float32),
            high=np.array([
                self.cfg.pi_bounds[1],
                self.cfg.u_bounds[1],
                self.cfg.g_bounds[1],
                self.cfg.r_bounds[1],
                self.cfg.d_bounds[1],
                self.cfg.E_pi_bounds[1],
                self.cfg.tau_bounds[1],
                self.cfg.G_bounds[1],
            ], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([
                self.cfg.delta_r_bounds[0],
                self.cfg.delta_G_bounds[0],
                self.cfg.delta_tau_bounds[0],
            ], dtype=np.float32),
            high=np.array([
                self.cfg.delta_r_bounds[1],
                self.cfg.delta_G_bounds[1],
                self.cfg.delta_tau_bounds[1],
            ], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        obs = self.economy.reset(seed=seed)
        info = {
            "step": self.economy.step_count,
            "tau": obs["tau"],
            "G_t": obs["G"],
        }
        return self._vectorize_obs(obs), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, got {action_array.shape}"
            )

        clipped_action = np.clip(action_array, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.economy.step(
            tuple(float(x) for x in clipped_action)
        )
        info["raw_action"] = action_array.copy()
        info["clipped_action"] = clipped_action.copy()

        return (
            self._vectorize_obs(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _vectorize_obs(self, obs: dict[str, float]) -> np.ndarray:
        return np.array([obs[key] for key in OBSERVATION_KEYS], dtype=np.float32)
