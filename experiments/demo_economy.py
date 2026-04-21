#!/usr/bin/env python3
"""
Standalone demo script:
    1. Simulate the economy under a simple Taylor-like rule.
    2. Save trajectory data & Matplotlib static plots.
    3. Render a Manim animation (optional, if manim is available).

Usage:
    source .venv/bin/activate
    python experiments/demo_economy.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.economy import Economy
from src.utils.config import EconomyConfig, RewardConfig


# --------------------------------------------------------------------------- #
#  1. Policy: simple Taylor-like rule + passive fiscal
# --------------------------------------------------------------------------- #

def taylor_policy(obs: dict, cfg: EconomyConfig) -> tuple:
    """
    Simple rule-based policy for demo purposes.
    Monetary:  delta_r = phi_pi * (pi - pi_star) + phi_u * (u - u_star)
    Fiscal:    delta_G = -psi_g * (g - g_star)  - psi_d * (d - d_star)
    """
    phi_pi, phi_u = 0.5, 0.2
    psi_g, psi_d = 0.10, 0.03

    delta_r = phi_pi * (obs["pi"] - cfg.pi_star) + phi_u * (obs["u"] - cfg.u_star)
    delta_G = -psi_g * (obs["g"] - cfg.g_star) - psi_d * (obs["d"] - cfg.d_star)

    # clip to action bounds
    delta_r = np.clip(delta_r, -2.5, 2.5)
    delta_G = np.clip(delta_G, -5.0, 5.0)
    return delta_r, delta_G


# --------------------------------------------------------------------------- #
#  2. Simulation
# --------------------------------------------------------------------------- #

def simulate(seed: int = 42, steps: int = 200, shock_scenario: str = "baseline") -> list:
    cfg = EconomyConfig(max_steps=steps)
    rwd = RewardConfig()
    econ = Economy(config=cfg, reward_config=rwd, seed=seed)

    obs = econ.reset(seed=seed)
    for t in range(steps):
        action = taylor_policy(obs, cfg)

        # shock scenarios
        if shock_scenario == "demand_shock" and 50 <= t < 70:
            eps_d = float(np.random.default_rng(seed + t).normal(0.0, 3.0))
            eps_s = None
        elif shock_scenario == "supply_shock" and 50 <= t < 70:
            eps_d = None
            eps_s = float(np.random.default_rng(seed + t).normal(0.0, 2.0))
        else:
            eps_d = eps_s = None

        obs, reward, done, info = econ.step(action, demand_shock=eps_d, supply_shock=eps_s)
        if done:
            break

    return econ.get_history()


# --------------------------------------------------------------------------- #
#  3. Static Matplotlib plots
# --------------------------------------------------------------------------- #

def plot_trajectory(history: list, out_dir: Path, title_suffix: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [h["step"] for h in history]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Macroeconomic Trajectory {title_suffix}", fontsize=14)

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

    plt.tight_layout()
    fname = out_dir / f"trajectory{title_suffix.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    print(f"Saved static plot: {fname}")
    plt.close(fig)

    # --- action plot ---
    fig, ax = plt.subplots(figsize=(10, 4))
    dr = [h.get("delta_r", 0.0) for h in history]
    dG = [h.get("delta_G", 0.0) for h in history]
    ax.plot(steps, dr, label="Δr (monetary)", color="purple")
    ax.plot(steps, dG, label="ΔG (fiscal)", color="teal")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Policy Action")
    ax.set_xlabel("Quarter")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Policy Actions {title_suffix}")
    fname2 = out_dir / f"actions{title_suffix.replace(' ', '_')}.png"
    fig.savefig(fname2, dpi=200)
    print(f"Saved action plot: {fname2}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  4. Manim scene (optional)
# --------------------------------------------------------------------------- #

MANIM_SCENE = '''
from manim import *
import json
from pathlib import Path

class EconomyDynamics(Scene):
    """
    Animate a single macroeconomic trajectory.
    Expects a JSON file `outputs/figures/demo_history.json` produced by
    the demo script.
    """

    def construct(self):
        json_path = Path("outputs/figures/demo_history.json")
        if not json_path.exists():
            self.add(Text("JSON not found — run demo_economy.py first").scale(0.5))
            return

        with open(json_path) as f:
            history = json.load(f)

        # Extract series
        steps = [h["step"] for h in history]
        pi = [h["pi"] for h in history]
        u  = [h["u"]  for h in history]
        g  = [h["g"]  for h in history]

        # Normalise to [-3, 3] for plotting on screen
        def norm(vals, lo=-5, hi=15):
            return [(v - lo) / (hi - lo) * 6 - 3 for v in vals]

        pi_n = norm(pi, -5, 15)
        u_n  = norm(u,  0,  20)
        g_n  = norm(g, -10, 15)

        axes = Axes(
            x_range=[0, len(steps), 50],
            y_range=[-3, 3, 1],
            axis_config={"include_tip": False},
            x_length=10,
            y_length=4,
        ).to_edge(DOWN)

        labels = axes.get_axis_labels(x_label="t", y_label="value")

        pi_dots = VGroup(*[
            Dot(axes.c2p(t, v), radius=0.03, color=RED)
            for t, v in zip(steps, pi_n)
        ])
        u_dots = VGroup(*[
            Dot(axes.c2p(t, v), radius=0.03, color=BLUE)
            for t, v in zip(steps, u_n)
        ])
        g_dots = VGroup(*[
            Dot(axes.c2p(t, v), radius=0.03, color=GREEN)
            for t, v in zip(steps, g_n)
        ])

        legend = VGroup(
            Dot(color=RED).scale(0.6),
            Text("π", font_size=20).next_to(Dot(color=RED), RIGHT),
            Dot(color=BLUE).scale(0.6).shift(DOWN*0.4),
            Text("u", font_size=20).next_to(Dot(color=BLUE).shift(DOWN*0.4), RIGHT),
            Dot(color=GREEN).scale(0.6).shift(DOWN*0.8),
            Text("g", font_size=20).next_to(Dot(color=GREEN).shift(DOWN*0.8), RIGHT),
        ).to_corner(UR)

        self.play(Create(axes), Write(labels))
        self.play(FadeIn(legend))

        # Animate trajectories by revealing progressively
        self.play(Create(pi_dots), run_time=4, rate_func=linear)
        self.play(Create(u_dots),  run_time=4, rate_func=linear)
        self.play(Create(g_dots),  run_time=4, rate_func=linear)

        self.wait(2)
'''


def write_manim_scene(out_dir: Path):
    scene_path = out_dir / "manim_scenes.py"
    scene_path.write_text(MANIM_SCENE)
    print(f"Wrote Manim scene template: {scene_path}")


# --------------------------------------------------------------------------- #
#  5. Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- baseline ---
    hist_base = simulate(seed=42, steps=200, shock_scenario="baseline")
    plot_trajectory(hist_base, out_dir, title_suffix="(Baseline)")

    # --- demand shock ---
    hist_d = simulate(seed=42, steps=200, shock_scenario="demand_shock")
    plot_trajectory(hist_d, out_dir, title_suffix="(Demand Shock)")

    # --- supply shock ---
    hist_s = simulate(seed=42, steps=200, shock_scenario="supply_shock")
    plot_trajectory(hist_s, out_dir, title_suffix="(Supply Shock)")

    # --- export JSON for Manim ---
    import json
    with open(out_dir / "demo_history.json", "w") as f:
        json.dump(hist_base, f)
    print(f"Exported JSON for Manim: {out_dir / 'demo_history.json'}")

    # --- write Manim scene template ---
    write_manim_scene(Path("viz"))

    print("\nDemo complete. Check outputs/figures/ for plots.")
    print("To render Manim scene:")
    print("    manim -pql viz/manim_scenes.py EconomyDynamics")
