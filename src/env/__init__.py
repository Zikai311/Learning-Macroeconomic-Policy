from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register, registry

from src.env.macro_env import MacroEnv
from src.env.wrappers import NormalizedActionWrapper

ENV_ID = "MacroEnv-v0"
NORMALIZED_ENV_ID = "MacroEnvNormalized-v0"


def make_macro_env(*, normalize_actions: bool = False, **kwargs):
    env = MacroEnv(**kwargs)
    if normalize_actions:
        env = NormalizedActionWrapper(env)
    return env


def make_normalized_macro_env(**kwargs):
    return make_macro_env(normalize_actions=True, **kwargs)


if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="src.env.macro_env:MacroEnv",
    )

if NORMALIZED_ENV_ID not in registry:
    register(
        id=NORMALIZED_ENV_ID,
        entry_point="src.env:make_normalized_macro_env",
    )


__all__ = [
    "ENV_ID",
    "NORMALIZED_ENV_ID",
    "MacroEnv",
    "NormalizedActionWrapper",
    "make_macro_env",
    "make_normalized_macro_env",
]
