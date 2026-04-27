"""
Hand-written linear baseline policy for the macro simulation.

This file only defines how a simple Taylor-like strategy turns the
current observed macro state into the next action.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from src.utils.config import EconomyConfig, TaylorPolicyConfig

Action = tuple[float, float, float]
ActionMaker = Callable[[Mapping[str, float]], Action]


def linear_taylor_action(
    obs: Mapping[str, float],
    cfg: EconomyConfig,
    policy_cfg: TaylorPolicyConfig,
) -> Action:
    """
    Linear Taylor-style rule with mild fiscal stabilization.
    """
    delta_r = (
        policy_cfg.phi_pi * (obs["pi"] - cfg.pi_star) +
        policy_cfg.phi_u * (obs["u"] - cfg.u_star)
    )
    delta_G = (
        -policy_cfg.psi_g * (obs["g"] - cfg.g_star) -
        policy_cfg.psi_d * (obs["d"] - cfg.d_star)
    )
    delta_tau = policy_cfg.psi_tau * (obs["d"] - cfg.d_star)

    return (
        float(np.clip(delta_r, *cfg.delta_r_bounds)),
        float(np.clip(delta_G, *cfg.delta_G_bounds)),
        float(np.clip(delta_tau, *cfg.delta_tau_bounds)),
    )


def build_action_maker(
    cfg: EconomyConfig,
    policy_cfg: TaylorPolicyConfig | None = None,
) -> ActionMaker:
    """
    Build a reusable action function for `run_economy.py`.

    Later RL policies can expose the same interface and be swapped in by
    changing the `--action-maker` argument.
    """
    policy_cfg = policy_cfg or TaylorPolicyConfig()

    def action_maker(obs: Mapping[str, float]) -> Action:
        return linear_taylor_action(obs, cfg, policy_cfg)

    return action_maker
