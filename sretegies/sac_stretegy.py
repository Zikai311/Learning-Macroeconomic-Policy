"""
Inference-time strategy wrapper for trained SAC models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.env.macro_env import OBSERVATION_KEYS
from src.utils.config import EconomyConfig

Action = tuple[float, float, float]
ActionMaker = Callable[[Mapping[str, float]], Action]

DEFAULT_MODEL_PATH = Path("outputs/models/sac_macro_long_best/best_model.zip")
FALLBACK_MODEL_PATH = Path("outputs/models/sac_macro_long.zip")
DEFAULT_METADATA_PATH = Path("outputs/models/sac_macro_long_metadata.json")


def build_action_maker(cfg: EconomyConfig) -> ActionMaker:
    """
    Build an action maker that loads a trained SAC checkpoint once and then
    returns actions compatible with `run_economy.py`.

    Override the default checkpoint with:
        SAC_MODEL_PATH=/path/to/model.zip
        SAC_METADATA_PATH=/path/to/model_metadata.json
    """
    model_path = Path(os.environ.get("SAC_MODEL_PATH", str(_default_model_path())))
    metadata_path = Path(os.environ.get("SAC_METADATA_PATH", str(DEFAULT_METADATA_PATH)))

    if not model_path.exists():
        raise FileNotFoundError(
            f"SAC model not found at {model_path}. Train one first or set SAC_MODEL_PATH."
        )

    trainer_cfg = _load_trainer_config(metadata_path)
    normalize_actions = bool(trainer_cfg.get("normalize_actions", True))
    deterministic = _parse_bool_env("SAC_DETERMINISTIC", default=True)
    model = SAC.load(str(model_path))

    def action_maker(obs: Mapping[str, float]) -> Action:
        obs_vec = np.array([obs[key] for key in OBSERVATION_KEYS], dtype=np.float32)
        action, _ = model.predict(obs_vec, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        if normalize_actions:
            action = _denormalize_action(action, cfg)

        clipped = np.array([
            np.clip(action[0], *cfg.delta_r_bounds),
            np.clip(action[1], *cfg.delta_G_bounds),
            np.clip(action[2], *cfg.delta_tau_bounds),
        ], dtype=np.float32)
        return float(clipped[0]), float(clipped[1]), float(clipped[2])

    return action_maker


def _denormalize_action(action: np.ndarray, cfg: EconomyConfig) -> np.ndarray:
    low = np.array([
        cfg.delta_r_bounds[0],
        cfg.delta_G_bounds[0],
        cfg.delta_tau_bounds[0],
    ], dtype=np.float32)
    high = np.array([
        cfg.delta_r_bounds[1],
        cfg.delta_G_bounds[1],
        cfg.delta_tau_bounds[1],
    ], dtype=np.float32)
    clipped = np.clip(action, -1.0, 1.0)
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0
    return bias + clipped * scale


def _load_trainer_config(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("trainer_config", {})


def _parse_bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _default_model_path() -> Path:
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return FALLBACK_MODEL_PATH
