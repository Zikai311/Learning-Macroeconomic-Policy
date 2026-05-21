"""
Minimal SAC training scaffold for the macro environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.env import make_macro_env
from src.rl.callbacks import EpisodeMacroMetricsCallback
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
    best_model_dir: str = "outputs/models/sac_best"
    checkpoint_dir: str = "outputs/models/checkpoints"
    eval_log_dir: str = "outputs/logs/eval"
    metrics_path: str = "outputs/logs/sac/episode_metrics.jsonl"
    run_name: str = "macro_sac"
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    checkpoint_freq: int = 10_000
    save_replay_buffer: bool = False
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


def build_callbacks(
    trainer_config: SACTrainerConfig,
    *,
    economy_config: EconomyConfig | None = None,
    reward_config: RewardConfig | None = None,
):
    callbacks = []
    auxiliary_envs = []

    callbacks.append(EpisodeMacroMetricsCallback(metrics_path=trainer_config.metrics_path))

    if trainer_config.checkpoint_freq > 0:
        checkpoint_dir = Path(trainer_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=trainer_config.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=trainer_config.run_name,
                save_replay_buffer=trainer_config.save_replay_buffer,
            )
        )

    if trainer_config.eval_freq > 0 and trainer_config.n_eval_episodes > 0:
        eval_env = build_training_env(
            economy_config=economy_config,
            reward_config=reward_config,
            normalize_actions=trainer_config.normalize_actions,
            seed=trainer_config.seed + 10_000,
        )
        auxiliary_envs.append(eval_env)
        best_model_dir = Path(trainer_config.best_model_dir)
        eval_log_dir = Path(trainer_config.eval_log_dir)
        best_model_dir.mkdir(parents=True, exist_ok=True)
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_log_dir),
                eval_freq=trainer_config.eval_freq,
                n_eval_episodes=trainer_config.n_eval_episodes,
                deterministic=True,
                render=False,
            )
        )

    return CallbackList(callbacks), auxiliary_envs


def save_run_metadata(
    trainer_config: SACTrainerConfig,
    *,
    economy_config: EconomyConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> str:
    model_path = Path(trainer_config.model_path)
    metadata_path = model_path.with_name(f"{model_path.name}_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "trainer_config": _to_jsonable(trainer_config),
        "economy_config": _to_jsonable(economy_config or EconomyConfig()),
        "reward_config": _to_jsonable(reward_config or RewardConfig()),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(metadata_path)


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
    callbacks, auxiliary_envs = build_callbacks(
        trainer_config,
        economy_config=economy_config,
        reward_config=reward_config,
    )
    metadata_path = save_run_metadata(
        trainer_config,
        economy_config=economy_config,
        reward_config=reward_config,
    )
    model.learn(
        total_timesteps=trainer_config.total_timesteps,
        callback=callbacks,
        progress_bar=trainer_config.progress_bar,
        tb_log_name=trainer_config.run_name,
    )

    model_path = Path(trainer_config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    if trainer_config.save_replay_buffer:
        replay_buffer_path = model_path.with_name(f"{model_path.name}_replay_buffer.pkl")
        model.save_replay_buffer(str(replay_buffer_path))
    env.close()
    for auxiliary_env in auxiliary_envs:
        auxiliary_env.close()
    print(f"Saved training metadata: {metadata_path}")
    return model, str(model_path)


def _to_jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    return value
