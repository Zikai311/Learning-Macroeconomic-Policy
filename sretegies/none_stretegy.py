"""
No-intervention baseline policy for the macro simulation.

This policy always returns zero changes, meaning the central bank and
government leave rates, spending, and taxes unchanged each step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from src.utils.config import EconomyConfig

Action = tuple[float, float, float]
ActionMaker = Callable[[Mapping[str, float]], Action]


def no_intervention_action(_obs: Mapping[str, float]) -> Action:
    """
    Return zero action in every policy dimension.
    """
    return 0.0, 0.0, 0.0


def build_action_maker(_cfg: EconomyConfig) -> ActionMaker:
    """
    Build a reusable no-intervention policy for `run_economy.py`.
    """
    return no_intervention_action
