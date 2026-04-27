"""
Minimal SAC training scaffold for the macro environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src.env import make_macro_env
from src.utils.config import EconomyConfig, RewardConfig


@dataclass
class SACTrainerConfig:
    total_timesteps: int = 200_000
    seed: int = 42
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    learning_starts: int = 1_000
    train_freq: int = 1
    gradient_steps: int = 1
    normalize_actions: bool = True
    tensorboard_log: str = "outputs/logs/sac"
    model_path: str = "outputs/models/sac_macro"
    run_name: str = "macro_sac"
    progress_bar: bool = False
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": {"pi": [256, 256, 128], "qf": [256, 256, 128]},
        "activation_fn": torch.nn.ReLU,
    })


def build_training_env(
    *,
    economy_config: EconomyConfig | None = None,
    reward_config: RewardConfig | None = None,
    normalize_actions: bool = True,
    seed: int | None = None,
):
    env = make_macro_env(
        config=economy_config,
        reward_config=reward_config,
        normalize_actions=normalize_actions,
    )
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_model(
    env,
    trainer_config: SACTrainerConfig,
) -> SAC:
    tensorboard_log = trainer_config.tensorboard_log
    if tensorboard_log and importlib.util.find_spec("tensorboard") is None:
        tensorboard_log = None

    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=trainer_config.learning_rate,
        buffer_size=trainer_config.buffer_size,
        batch_size=trainer_config.batch_size,
        gamma=trainer_config.gamma,
        tau=trainer_config.tau,
        learning_starts=trainer_config.learning_starts,
        train_freq=trainer_config.train_freq,
        gradient_steps=trainer_config.gradient_steps,
        tensorboard_log=tensorboard_log,
        policy_kwargs=trainer_config.policy_kwargs,
        seed=trainer_config.seed,
        device="auto",
        verbose=1,
    )


def train_sac(
    trainer_config: SACTrainerConfig,
    *,
    economy_config: EconomyConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> tuple[SAC, str]:
    env = build_training_env(
        economy_config=economy_config,
        reward_config=reward_config,
        normalize_actions=trainer_config.normalize_actions,
        seed=trainer_config.seed,
    )
    model = build_model(env, trainer_config)
    model.learn(
        total_timesteps=trainer_config.total_timesteps,
        progress_bar=trainer_config.progress_bar,
        tb_log_name=trainer_config.run_name,
    )

    model_path = Path(trainer_config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    env.close()
    return model, str(model_path)
