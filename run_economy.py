#!/usr/bin/env python3
"""
Run the macroeconomic simulation with a pluggable action maker.

Examples
--------
python run_economy.py
python run_economy.py --shock-scenario demand_shock
python run_economy.py --action-maker sretegies.linear_stretegy:build_action_maker
python run_economy.py --action-maker sretegies.none_stretegy:build_action_maker
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.models.economy import Economy
from src.utils.config import EconomyConfig, RewardConfig

Action = tuple[float, float, float]
ActionMaker = Callable[[Mapping[str, float]], Action]


def load_action_maker(spec: str, cfg: EconomyConfig) -> ActionMaker:
    """
    Load an action-maker builder from `module:function` and bind it to the config.
    """
    if ":" not in spec:
        raise ValueError("action-maker must use the format 'module:function'")

    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    builder = getattr(module, attr_name)
    action_maker = builder(cfg)

    if not callable(action_maker):
        raise TypeError(f"{spec} did not return a callable action maker")

    return action_maker


def sample_shocks(step: int, shock_scenario: str, seed: int) -> tuple[float | None, float | None]:
    """
    Optionally inject larger structured shocks on top of the baseline noise.
    """
    if shock_scenario == "demand_shock" and 50 <= step < 70:
        return float(np.random.default_rng(seed + step).normal(0.0, 3.0)), None
    if shock_scenario == "supply_shock" and 50 <= step < 70:
        return None, float(np.random.default_rng(seed + step).normal(0.0, 2.0))
    return None, None


def simulate(
    action_maker: ActionMaker,
    *,
    seed: int = 42,
    steps: int | None = None,
    shock_scenario: str = "baseline",
    config: EconomyConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> list[dict]:
    """
    Roll the economy forward using any action maker with signature `action(obs)`.
    """
    cfg = config or EconomyConfig()
    if steps is None:
        steps = cfg.max_steps
    elif cfg.max_steps != steps:
        cfg = replace(cfg, max_steps=steps)

    econ = Economy(config=cfg, reward_config=reward_config or RewardConfig(), seed=seed)

    obs = econ.reset(seed=seed)
    for step in range(steps):
        action = action_maker(obs)
        eps_d, eps_s = sample_shocks(step, shock_scenario, seed)

        obs, reward, terminated, truncated, info = econ.step(
            action,
            demand_shock=eps_d,
            supply_shock=eps_s,
        )
        if terminated or truncated:
            break

    return econ.get_history()


def save_history(history: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the macroeconomic simulator.")
    parser.add_argument(
        "--action-maker",
        default="sretegies.linear_stretegy:build_action_maker",
        help="Import path to a builder returning an action callable.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reset and structured shocks.")
    parser.add_argument("--steps", type=int, default=200, help="Number of quarters to simulate.")
    parser.add_argument(
        "--shock-scenario",
        choices=("baseline", "demand_shock", "supply_shock"),
        default="baseline",
        help="Structured shock scenario to layer on top of baseline noise.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures",
        help="Directory for saved charts and JSON history.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Label used in filenames and chart titles. Defaults to the scenario name.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Matplotlib plots and only save the raw JSON history.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = EconomyConfig(max_steps=args.steps)
    action_maker = load_action_maker(args.action_maker, cfg)
    history = simulate(
        action_maker,
        seed=args.seed,
        steps=args.steps,
        shock_scenario=args.shock_scenario,
        config=cfg,
    )

    run_name = args.run_name or args.shock_scenario.replace("_", " ").title()
    slug = run_name.lower().replace(" ", "_")
    out_dir = Path(args.output_dir)
    history_path = save_history(history, out_dir / f"history_{slug}.json")

    print(f"Saved history: {history_path}")

    if not args.no_plots:
        from src.utils.plotting import plot_trajectory

        plot_paths = plot_trajectory(history, out_dir, run_name=run_name)
        print(f"Saved trajectory plot: {plot_paths['trajectory']}")
        print(f"Saved policy plot: {plot_paths['actions']}")


if __name__ == "__main__":
    main()
