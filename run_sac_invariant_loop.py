#!/usr/bin/env python3
"""
Run the trained SAC policy from an initial state that lies off the controllable
hyperplane of the linearized macro system.

The paper (sections "Kalman / Lyapunov interpretation of the linear example")
identifies a structural invariant of the controlled linear system. The paper
writes it as q = eta * tilde_u + gamma * tilde_e with eta = rho*beta/(1-rho),
based on a contemporaneous Phillips equation pi_t = E_t pi_{t+1} + beta * g_t.

The implementation in src/models/economy.py uses *lagged* expectations:
    pi_t = E_pi_{t-1} + beta * (g_t - g_star),
    E_pi_t = rho * pi_t + (1-rho) * E_pi_{t-1}.
Working through the same algebra for that timing convention gives
    Delta tilde_e_t = rho*beta * tilde_g_t,   Delta tilde_u_t = -gamma * tilde_g_t,
so the structural invariant of the code is

    q_t = gamma * (u_t - u*) + rho*beta * (E_pi_t - pi*).

Same Kalman/Lyapunov mechanism as the paper (w^T A = w^T, w^T B = 0, hence
1 is a left eigenvalue of every closed-loop G = A + BK and rho(G) >= 1).

This script makes that prediction visible empirically: shocks are switched off,
the system is initialized with q_0 != 0, and the SAC policy is rolled out. The
saved figure plots the macro trajectory together with q_t and ||s_t - s*||^2.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.economy import Economy
from src.utils.config import EconomyConfig, RewardConfig
from sretegies.sac_stretegy import build_action_maker


def main() -> None:
    cfg = EconomyConfig(sigma_d=0.0, sigma_s=0.0, max_steps=200)
    rwd = RewardConfig()

    action_maker = build_action_maker(cfg)

    econ = Economy(config=cfg, reward_config=rwd, seed=0)
    econ.reset(seed=0)

    # Pin every variable to the analytical steady state, then perturb only
    # (u, E_pi). This guarantees q_0 != 0 while leaving all other states clean.
    econ.pi = 4.0
    econ.u = 8.0
    econ.g = cfg.g_star
    econ.r = cfg.r_0
    econ.d = cfg.d_star
    econ.E_pi = 4.0
    econ.tau = cfg.tau

    econ.G_lag = cfg.G_0
    econ.r_lag = econ.r
    econ.E_pi_lag = econ.E_pi

    econ.history.clear()
    econ._record(econ._obs_dict(), action=None, shocks=None, reward=0.0)

    a_u = cfg.gamma
    a_e = cfg.rho * cfg.beta

    def invariant(u: float, e_pi: float) -> float:
        return a_u * (u - cfg.u_star) + a_e * (e_pi - cfg.pi_star)

    q_series = [invariant(econ.u, econ.E_pi)]

    for _ in range(cfg.max_steps):
        obs = econ._obs_dict()
        action = action_maker(obs)
        _, _, terminated, truncated, _ = econ.step(action)
        q_series.append(invariant(econ.u, econ.E_pi))
        if terminated or truncated:
            break

    history = econ.get_history()
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = [h["step"] for h in history]
    panels = [
        ("pi", "Inflation π (%)", "red", cfg.pi_star),
        ("u", "Unemployment u (%)", "blue", cfg.u_star),
        ("g", "Growth g (%)", "green", cfg.g_star),
        ("r", "Interest rate r (%)", "purple", None),
        ("d", "Debt/GDP d (%)", "orange", cfg.d_star),
        ("E_pi", "Expected inflation Eπ (%)", "brown", cfg.pi_star),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle("SAC rollout from non-zero invariant (shocks off)", fontsize=14)

    for ax, (key, label, color, target) in zip(axes.flat[:6], panels):
        vals = [h[key] for h in history]
        ax.plot(steps, vals, color=color, lw=1.2)
        if target is not None:
            ax.axhline(target, ls="--", color="gray", lw=0.8, label=f"target = {target}")
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel(label)
        ax.set_xlabel("Quarter")
        ax.grid(True, alpha=0.3)

    ax_q = axes[3, 0]
    q_steps = list(range(len(q_series)))
    ax_q.plot(q_steps, q_series, color="black", lw=1.2, label="q_t")
    ax_q.axhline(q_series[0], ls="--", color="red", lw=0.8, label=f"q_0 = {q_series[0]:.3f}")
    ax_q.set_ylabel("Invariant q_t")
    ax_q.set_xlabel("Quarter")
    ax_q.set_title("q_t = γ·(u−u*) + ρβ·(Eπ−π*)")
    ax_q.legend(loc="best", fontsize=8)
    ax_q.grid(True, alpha=0.3)

    ax_err = axes[3, 1]
    err = [
        (h["pi"] - cfg.pi_star) ** 2
        + (h["u"] - cfg.u_star) ** 2
        + (h["g"] - cfg.g_star) ** 2
        + (h["E_pi"] - cfg.pi_star) ** 2
        for h in history
    ]
    ax_err.plot(steps, err, color="darkred", lw=1.2)
    ax_err.set_ylabel("‖s_t − s*‖² (key states)")
    ax_err.set_xlabel("Quarter")
    ax_err.set_title("Distance to steady state")
    ax_err.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig_path = out_dir / "sac_invariant_loop.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    history_path = out_dir / "history_sac_invariant_loop.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"a_u (γ) = {a_u:.4f},  a_e (ρβ) = {a_e:.4f}")
    print(f"q_0   = {q_series[0]:.4f}")
    print(f"q_end = {q_series[-1]:.4f}")
    print(f"q range = [{min(q_series):.4f}, {max(q_series):.4f}]")
    print(f"final residual  = {err[-1]:.4f}")
    print(f"saved figure  : {fig_path}")
    print(f"saved history : {history_path}")


if __name__ == "__main__":
    main()
