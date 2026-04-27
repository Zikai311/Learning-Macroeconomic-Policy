"""
Reusable Gymnasium wrappers for the macro environment.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizedActionWrapper(gym.ActionWrapper):
    """
    Map a Box action space from [-1, 1] into the environment's native bounds.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("NormalizedActionWrapper requires a Box action space")

        self._actual_low = env.action_space.low.astype(np.float32)
        self._actual_high = env.action_space.high.astype(np.float32)
        self.action_space = spaces.Box(
            low=-np.ones_like(self._actual_low, dtype=np.float32),
            high=np.ones_like(self._actual_high, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32)
        clipped = np.clip(normalized, self.action_space.low, self.action_space.high)
        scale = (self._actual_high - self._actual_low) / 2.0
        bias = (self._actual_high + self._actual_low) / 2.0
        return bias + clipped * scale

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        actual = np.asarray(action, dtype=np.float32)
        scale = (self._actual_high - self._actual_low) / 2.0
        bias = (self._actual_high + self._actual_low) / 2.0
        return (actual - bias) / scale

    def step(self, action):
        normalized = np.asarray(action, dtype=np.float32)
        clipped = np.clip(normalized, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(self.action(clipped))
        info = dict(info)
        info["normalized_action"] = clipped.copy()
        return obs, reward, terminated, truncated, info
