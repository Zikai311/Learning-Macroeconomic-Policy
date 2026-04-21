
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
