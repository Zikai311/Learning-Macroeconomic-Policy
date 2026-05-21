#!/usr/bin/env python3
"""
CLI entry point for training a SAC agent on the macro environment.
"""

from __future__ import annotations

import argparse

from src.rl import SACTrainerConfig, train_sac
from src.utils.config import EconomyConfig, RewardConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC on MacroEnv.")
    parser.add_argument("--total-timesteps", type=int, default=50_000, help="Number of training timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="SAC learning rate.")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Replay buffer size.")
    parser.add_argument("--batch-size", type=int, default=256, help="SAC batch size.")
    parser.add_argument("--learning-starts", type=int, default=1_000, help="Warmup steps before gradient updates.")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency in training steps.")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Episodes per evaluation pass.")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000, help="Checkpoint frequency in training steps.")
    parser.add_argument(
        "--normalize-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the environment so SAC acts in [-1, 1]^3.",
    )
    parser.add_argument("--tensorboard-log", default="outputs/logs/sac", help="TensorBoard log directory.")
    parser.add_argument("--model-path", default="outputs/models/sac_macro", help="Path prefix for saved model.")
    parser.add_argument("--best-model-dir", default="outputs/models/sac_best", help="Directory for best eval model.")
    parser.add_argument("--checkpoint-dir", default="outputs/models/checkpoints", help="Directory for periodic checkpoints.")
    parser.add_argument("--eval-log-dir", default="outputs/logs/eval", help="Directory for evaluation logs.")
    parser.add_argument("--metrics-path", default="outputs/logs/sac/episode_metrics.jsonl", help="JSONL file for episode macro metrics.")
    parser.add_argument("--run-name", default="macro_sac", help="TensorBoard run name.")
    parser.add_argument(
        "--save-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the replay buffer after training.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show tqdm progress bar during learning.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trainer_config = SACTrainerConfig(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        normalize_actions=args.normalize_actions,
        tensorboard_log=args.tensorboard_log,
        model_path=args.model_path,
        best_model_dir=args.best_model_dir,
        checkpoint_dir=args.checkpoint_dir,
        eval_log_dir=args.eval_log_dir,
        metrics_path=args.metrics_path,
        run_name=args.run_name,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        progress_bar=args.progress_bar,
    )
    _, model_path = train_sac(
        trainer_config,
        economy_config=EconomyConfig(),
        reward_config=RewardConfig(),
    )
    print(f"Saved SAC model: {model_path}")


if __name__ == "__main__":
    main()
