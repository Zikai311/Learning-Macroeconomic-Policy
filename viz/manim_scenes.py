
from manim import *
import json
from pathlib import Path

class EconomyDynamics(Scene):
    """
    Animate a single macroeconomic trajectory with connected line paths.
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

        # --- Build connected line paths (VMobject) for each series ---
        def build_path(data, color):
            """Create a VMobject path through all points."""
            path = VMobject(stroke_color=color, stroke_width=2)
            # Convert first point to Manim coordinates
            points = [axes.c2p(t, v) for t, v in zip(steps, data)]
            path.set_points_as_corners(points)
            return path

        pi_path = build_path(pi_n, RED)
        u_path  = build_path(u_n,  BLUE)
        g_path  = build_path(g_n,  GREEN)

        # --- Legend ---
        legend = VGroup(
            Dot(color=RED).scale(0.6),
            Text("π", font_size=20).next_to(Dot(color=RED), RIGHT),
            Dot(color=BLUE).scale(0.6).shift(DOWN*0.4),
            Text("u", font_size=20).next_to(Dot(color=BLUE).shift(DOWN*0.4), RIGHT),
            Dot(color=GREEN).scale(0.6).shift(DOWN*0.8),
            Text("g", font_size=20).next_to(Dot(color=GREEN).shift(DOWN*0.8), RIGHT),
        ).to_corner(UR)

        # --- Animate ---
        self.play(Create(axes), Write(labels))
        self.play(FadeIn(legend))

        # Draw each trajectory progressively
        self.play(Create(pi_path), run_time=4, rate_func=linear)
        self.play(Create(u_path),  run_time=4, rate_func=linear)
        self.play(Create(g_path),  run_time=4, rate_func=linear)

        self.wait(2)
