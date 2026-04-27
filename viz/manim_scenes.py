
from manim import *
import json
from pathlib import Path

class EconomyDynamics(Scene):
    """
    Animate a single macroeconomic trajectory.
    Expects a JSON file `outputs/figures/history_baseline.json` produced by
    `run_economy.py`.
    """

    def construct(self):
        json_path = Path("outputs/figures/history_baseline.json")
        if not json_path.exists():
            self.add(Text("JSON not found — run run_economy.py first").scale(0.5))
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
