import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_economy import load_action_maker, simulate
from sretegies.sac_stretegy import _denormalize_action
from src.utils.config import EconomyConfig


class RunEconomyTests(unittest.TestCase):
    def test_load_action_maker_builds_callable(self):
        action_maker = load_action_maker(
            "sretegies.linear_stretegy:build_action_maker",
            EconomyConfig(),
        )

        action = action_maker({
            "pi": 2.0,
            "u": 5.0,
            "g": 2.5,
            "r": 3.0,
            "d": 60.0,
            "E_pi": 2.0,
            "tau": 20.0,
        })

        self.assertEqual(len(action), 3)

    def test_load_none_strategy_returns_zero_action(self):
        action_maker = load_action_maker(
            "sretegies.none_stretegy:build_action_maker",
            EconomyConfig(),
        )

        action = action_maker({
            "pi": 4.0,
            "u": 7.0,
            "g": 1.0,
            "r": 5.0,
            "d": 80.0,
            "E_pi": 3.5,
            "tau": 22.0,
        })

        self.assertEqual(action, (0.0, 0.0, 0.0))

    def test_simulate_accepts_external_action_maker(self):
        def zero_policy(_obs):
            return 0.0, 0.0, 0.0

        history = simulate(
            zero_policy,
            seed=0,
            steps=3,
            shock_scenario="baseline",
            config=EconomyConfig(max_steps=3),
        )

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["step"], 0)
        self.assertEqual(history[-1]["step"], 3)

    def test_denormalize_action_maps_minus_one_to_low_and_plus_one_to_high(self):
        cfg = EconomyConfig()
        actual = _denormalize_action(np.array([-1.0, 1.0, 0.0], dtype=np.float32), cfg)

        np.testing.assert_allclose(
            actual,
            np.array([
                cfg.delta_r_bounds[0],
                cfg.delta_G_bounds[1],
                0.0,
            ], dtype=np.float32),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
