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
    parser.add_argument(
        "--normalize-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the environment so SAC acts in [-1, 1]^3.",
    )
    parser.add_argument("--tensorboard-log", default="outputs/logs/sac", help="TensorBoard log directory.")
    parser.add_argument("--model-path", default="outputs/models/sac_macro", help="Path prefix for saved model.")
    parser.add_argument("--run-name", default="macro_sac", help="TensorBoard run name.")
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
        normalize_actions=args.normalize_actions,
        tensorboard_log=args.tensorboard_log,
        model_path=args.model_path,
        run_name=args.run_name,
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
