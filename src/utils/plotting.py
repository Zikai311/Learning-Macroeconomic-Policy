"""
Reusable plotting helpers for macroeconomic simulation runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return slug or "run"


def plot_trajectory(history: list[dict], out_dir: Path, run_name: str = "Baseline") -> dict[str, Path]:
    """
    Save the standard state and policy charts for a simulation history.
    """
    if not history:
        raise ValueError("history must contain at least one recorded step")

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [h["step"] for h in history]
    slug = _slugify(run_name)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Macroeconomic Trajectory ({run_name})", fontsize=14)

    vars_and_labels = [
        ("pi", "Inflation π (%)", "red"),
        ("u", "Unemployment u (%)", "blue"),
        ("g", "GDP Growth g (%)", "green"),
        ("r", "Interest Rate r (%)", "purple"),
        ("d", "Debt/GDP d (%)", "orange"),
        ("E_pi", "Expected Inflation Eπ (%)", "brown"),
    ]

    for ax, (key, label, color) in zip(axes.flat, vars_and_labels):
        vals = [h[key] for h in history]
        ax.plot(steps, vals, color=color, lw=1.2)
        ax.set_ylabel(label)
        ax.set_xlabel("Quarter")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    trajectory_path = out_dir / f"trajectory_{slug}.png"
    fig.savefig(trajectory_path, dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    dr = [h.get("delta_r", 0.0) for h in history]
    dG = [h.get("delta_G", 0.0) for h in history]
    dtau = [h.get("delta_tau", 0.0) for h in history]
    ax.plot(steps, dr, label="Δr (monetary)", color="purple")
    ax.plot(steps, dG, label="ΔG (fiscal)", color="teal")
    ax.plot(steps, dtau, label="Δτ (tax)", color="crimson", ls="--")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Policy Action")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Policy Actions ({run_name})")

    ax2 = axes[1]
    rewards = [h.get("reward", 0.0) for h in history]
    ax2.plot(steps, rewards, color="darkgreen", lw=1.2, label="Reward")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.fill_between(
        steps,
        rewards,
        0,
        where=[reward >= 0 for reward in rewards],
        color="green",
        alpha=0.15,
        interpolate=True,
    )
    ax2.fill_between(
        steps,
        rewards,
        0,
        where=[reward < 0 for reward in rewards],
        color="red",
        alpha=0.15,
        interpolate=True,
    )
    ax2.set_ylabel("Economic Welfare (Reward)")
    ax2.set_xlabel("Quarter")
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Economic Health Over Time ({run_name})")

    fig.tight_layout()
    actions_path = out_dir / f"actions_{slug}.png"
    fig.savefig(actions_path, dpi=200)
    plt.close(fig)

    return {
        "trajectory": trajectory_path,
        "actions": actions_path,
    }
