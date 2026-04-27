import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.economy import Economy
from src.utils.config import EconomyConfig, RewardConfig


class EconomyDynamicsTests(unittest.TestCase):
    def test_reward_uses_configured_targets(self):
        reward_cfg = RewardConfig()

        reward_at_target = reward_cfg.compute(
            pi=3.0,
            u=6.0,
            g=4.0,
            d=80.0,
            pi_star=3.0,
            u_star=6.0,
            g_star=4.0,
            d_star=80.0,
        )
        reward_off_target = reward_cfg.compute(
            pi=3.0,
            u=6.0,
            g=4.0,
            d=80.0,
            pi_star=2.0,
            u_star=5.0,
            g_star=2.5,
            d_star=60.0,
        )

        self.assertEqual(reward_at_target, 0.0)
        self.assertLess(reward_off_target, reward_at_target)

    def test_step_matches_deterministic_transition_equations(self):
        cfg = EconomyConfig(
            alpha_1=0.2,
            alpha_2=0.1,
            beta=0.5,
            gamma=0.25,
            rho=0.4,
            pi_0=2.0,
            u_0=5.0,
            g_0=2.5,
            r_0=3.0,
            d_0=60.0,
            E_pi_0=2.0,
            G_0=20.0,
            tau=20.0,
            max_steps=10,
        )
        econ = Economy(config=cfg, reward_config=RewardConfig(), seed=123)
        pre = econ.reset(seed=123)

        obs, reward, terminated, truncated, info = econ.step(
            (1.0, 2.0, -1.0),
            demand_shock=0.0,
            supply_shock=0.0,
        )

        expected_g = cfg.alpha_1 * pre["G"] - cfg.alpha_2 * pre["r"]
        expected_pi = pre["E_pi"] + cfg.beta * (expected_g - cfg.g_star)
        expected_e_pi = cfg.rho * expected_pi + (1.0 - cfg.rho) * pre["E_pi"]
        expected_u = pre["u"] - cfg.gamma * (expected_g - cfg.g_star)
        expected_tau = pre["tau"] - 1.0
        expected_d = pre["d"] + (pre["G"] + 2.0) - expected_tau

        self.assertTrue(math.isclose(obs["r"], pre["r"] + 1.0, rel_tol=0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(obs["g"], expected_g, rel_tol=0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(obs["pi"], expected_pi, rel_tol=0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(obs["E_pi"], expected_e_pi, rel_tol=0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(obs["u"], expected_u, rel_tol=0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(obs["d"], expected_d, rel_tol=0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(obs["tau"], expected_tau, rel_tol=0.0, abs_tol=1e-9))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("terminated", info)
        self.assertIn("truncated", info)
        expected_reward = (
            -(expected_pi - cfg.pi_star) ** 2
            - (expected_u - cfg.u_star) ** 2
            + (expected_g - cfg.g_star)
            - 0.05 * (expected_d - cfg.d_star)
        )
        self.assertTrue(math.isclose(reward, expected_reward, rel_tol=0.0, abs_tol=1e-9))

    def test_horizon_sets_truncated_without_termination(self):
        cfg = EconomyConfig(max_steps=1, debt_limit=500.0)
        econ = Economy(config=cfg, reward_config=RewardConfig(), seed=0)
        econ.reset(seed=0)

        _, _, terminated, truncated, _ = econ.step(
            (0.0, 0.0, 0.0),
            demand_shock=0.0,
            supply_shock=0.0,
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_debt_limit_sets_termination(self):
        cfg = EconomyConfig(d_0=149.0, debt_limit=150.0, max_steps=10)
        econ = Economy(config=cfg, reward_config=RewardConfig(), seed=0)
        econ.reset(seed=0)

        _, _, terminated, truncated, _ = econ.step(
            (0.0, 5.0, -2.0),
            demand_shock=0.0,
            supply_shock=0.0,
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_history_records_tax_action_and_initial_state(self):
        cfg = EconomyConfig(max_steps=5)
        econ = Economy(config=cfg, reward_config=RewardConfig(), seed=1)
        econ.reset(seed=1)

        history = econ.get_history()
        self.assertEqual(len(history), 1)
        self.assertIn("tau", history[0])
        self.assertIn("G", history[0])
        self.assertEqual(history[0]["reward"], 0.0)

        econ.step((0.5, -1.0, 0.25), demand_shock=0.0, supply_shock=0.0)
        latest = econ.get_history()[-1]

        self.assertEqual(latest["delta_r"], 0.5)
        self.assertEqual(latest["delta_G"], -1.0)
        self.assertEqual(latest["delta_tau"], 0.25)
        self.assertIn("eps_d", latest)
        self.assertIn("eps_s", latest)


if __name__ == "__main__":
    unittest.main()
