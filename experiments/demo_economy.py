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
from src.utils.config import EconomyConfig, RewardConfig, TaylorPolicyConfig


# --------------------------------------------------------------------------- #
#  1. Policy: simple Taylor-like rule + passive fiscal
# --------------------------------------------------------------------------- #

def taylor_policy(obs: dict, cfg: EconomyConfig) -> tuple:
    """
    Simple rule-based policy for demo purposes.
    Monetary:  delta_r   = phi_pi * (pi - pi_star) + phi_u * (u - u_star)
    Fiscal:    delta_G   = -psi_g * (g - g_star)  - psi_d * (d - d_star)
               delta_tau = +psi_tau * (d - d_star)      (raise taxes when debt is high)
    """
    policy_cfg = TaylorPolicyConfig()

    delta_r = policy_cfg.phi_pi * (obs["pi"] - cfg.pi_star) + policy_cfg.phi_u * (obs["u"] - cfg.u_star)
    delta_G = -policy_cfg.psi_g * (obs["g"] - cfg.g_star) - policy_cfg.psi_d * (obs["d"] - cfg.d_star)
    delta_tau = policy_cfg.psi_tau * (obs["d"] - cfg.d_star)

    # clip to action bounds
    delta_r = np.clip(delta_r, *policy_cfg.delta_r_bounds)
    delta_G = np.clip(delta_G, *policy_cfg.delta_G_bounds)
    delta_tau = np.clip(delta_tau, *policy_cfg.delta_tau_bounds)
    return delta_r, delta_G, delta_tau


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

        obs, reward, terminated, truncated, info = econ.step(
            action,
            demand_shock=eps_d,
            supply_shock=eps_s,
        )
        if terminated or truncated:
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

    # --- action + reward plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1]})

    # Top: policy actions
    ax = axes[0]
    dr = [h.get("delta_r", 0.0) for h in history]
    dG = [h.get("delta_G", 0.0) for h in history]
    dt = [h.get("delta_tau", 0.0) for h in history]
    ax.plot(steps, dr, label="Δr (monetary)", color="purple")
    ax.plot(steps, dG, label="ΔG (fiscal)", color="teal")
    ax.plot(steps, dt, label="Δτ (tax)", color="crimson", ls="--")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Policy Action")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Policy Actions {title_suffix}")

    # Bottom: reward (economic health)
    ax2 = axes[1]
    rewards = [h.get("reward", 0.0) for h in history]
    ax2.plot(steps, rewards, color="darkgreen", lw=1.2, label="Reward")
    ax2.axhline(0, color="black", lw=0.5)
    # Shade positive / negative regions for intuitive reading
    ax2.fill_between(steps, rewards, 0, where=[r >= 0 for r in rewards],
                     color="green", alpha=0.15, interpolate=True)
    ax2.fill_between(steps, rewards, 0, where=[r < 0 for r in rewards],
                     color="red", alpha=0.15, interpolate=True)
    ax2.set_ylabel("Economic Welfare (Reward)")
    ax2.set_xlabel("Quarter")
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Economic Health Over Time {title_suffix}")

    plt.tight_layout()
    fname2 = out_dir / f"actions{title_suffix.replace(' ', '_')}.png"
    fig.savefig(fname2, dpi=200)
    print(f"Saved action + reward plot: {fname2}")
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

        # Build connected-line paths instead of scattered dots
        def make_path(values, color):
            path = VMobject(stroke_color=color, stroke_width=2, stroke_opacity=0.9)
            pts = [axes.c2p(t, v) for t, v in zip(steps, values)]
            path.set_points_as_corners(pts)
            return path

        pi_path = make_path(pi_n, RED)
        u_path = make_path(u_n, BLUE)
        g_path = make_path(g_n, GREEN)

        legend = VGroup(
            Line(ORIGIN, RIGHT*0.3, color=RED, stroke_width=3),
            Text("π", font_size=20).next_to(Line(ORIGIN, RIGHT*0.3, color=RED), RIGHT),
            Line(ORIGIN, RIGHT*0.3, color=BLUE, stroke_width=3).shift(DOWN*0.4),
            Text("u", font_size=20).next_to(Line(ORIGIN, RIGHT*0.3, color=BLUE).shift(DOWN*0.4), RIGHT),
            Line(ORIGIN, RIGHT*0.3, color=GREEN, stroke_width=3).shift(DOWN*0.8),
            Text("g", font_size=20).next_to(Line(ORIGIN, RIGHT*0.3, color=GREEN).shift(DOWN*0.8), RIGHT),
        ).to_corner(UR)

        self.play(Create(axes), Write(labels))
        self.play(FadeIn(legend))

        self.play(Create(pi_path), run_time=4, rate_func=linear)
        self.play(Create(u_path),  run_time=4, rate_func=linear)
        self.play(Create(g_path),  run_time=4, rate_func=linear)

        self.wait(2)
'''


def write_manim_scene(out_dir: Path):
    scene_path = out_dir / "manim_scenes.py"
    if scene_path.exists():
        print(f"Kept existing Manim scene: {scene_path}")
        return
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
