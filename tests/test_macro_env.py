import math
import os
import sys
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.env import MacroEnv
from src.utils.config import EconomyConfig, RewardConfig


class MacroEnvTests(unittest.TestCase):
    def test_reset_returns_expected_observation_vector(self):
        env = MacroEnv(config=EconomyConfig(max_steps=5), reward_config=RewardConfig())

        obs, info = env.reset(seed=123)

        self.assertEqual(obs.shape, (8,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertIn("G_t", info)
        self.assertIn("tau", info)
        self.assertTrue(math.isclose(float(obs[6]), info["tau"], rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(obs[7]), info["G_t"], rel_tol=0.0, abs_tol=1e-6))

    def test_step_matches_deterministic_transition_equations(self):
        cfg = EconomyConfig(
            alpha_1=0.2,
            alpha_2=0.1,
            beta=0.5,
            gamma=0.25,
            rho=0.4,
            sigma_d=0.0,
            sigma_s=0.0,
            max_steps=5,
        )
        env = MacroEnv(config=cfg, reward_config=RewardConfig())
        pre, _ = env.reset(seed=123)

        obs, reward, terminated, truncated, info = env.step(
            np.array([0.5, 2.0, -1.0], dtype=np.float32)
        )

        pre_pi, pre_u, _pre_g, pre_r, pre_d, pre_E_pi, pre_tau, pre_G = pre
        expected_r = min(max(pre_r + 0.5, cfg.r_bounds[0]), cfg.r_bounds[1])
        expected_g = cfg.alpha_1 * pre_G - cfg.alpha_2 * pre_r
        expected_pi = pre_E_pi + cfg.beta * (expected_g - cfg.g_star)
        expected_e_pi = cfg.rho * expected_pi + (1.0 - cfg.rho) * pre_E_pi
        expected_u = pre_u - cfg.gamma * (expected_g - cfg.g_star)
        expected_tau = pre_tau - 1.0
        expected_G = pre_G + 2.0
        expected_d = pre_d + expected_G - expected_tau

        self.assertTrue(math.isclose(float(obs[0]), expected_pi, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(obs[1]), expected_u, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(obs[2]), expected_g, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(obs[3]), expected_r, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(obs[4]), expected_d, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(obs[5]), expected_e_pi, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(obs[6]), expected_tau, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(obs[7]), expected_G, rel_tol=0.0, abs_tol=1e-6))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("clipped_action", info)
        np.testing.assert_allclose(
            info["clipped_action"],
            np.array([0.5, 2.0, -1.0], dtype=np.float32),
        )
        self.assertIsInstance(reward, float)

    def test_action_clipping_uses_shared_economy_bounds(self):
        cfg = EconomyConfig(max_steps=5)
        env = MacroEnv(config=cfg, reward_config=RewardConfig())
        env.reset(seed=0)

        obs, _, _, _, info = env.step(np.array([99.0, -99.0, 99.0], dtype=np.float32))

        np.testing.assert_allclose(
            info["clipped_action"],
            np.array([
                cfg.delta_r_bounds[1],
                cfg.delta_G_bounds[0],
                cfg.delta_tau_bounds[1],
            ], dtype=np.float32),
        )
        self.assertTrue(env.observation_space.contains(obs))

    def test_horizon_sets_truncated_without_termination(self):
        env = MacroEnv(config=EconomyConfig(max_steps=1, debt_limit=500.0), reward_config=RewardConfig())
        env.reset(seed=0)

        _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_debt_limit_sets_termination(self):
        env = MacroEnv(
            config=EconomyConfig(d_0=149.0, debt_limit=150.0, max_steps=10),
            reward_config=RewardConfig(),
        )
        env.reset(seed=0)

        _, _, terminated, truncated, _ = env.step(np.array([0.0, 2.0, -2.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_env_passes_gymnasium_checker(self):
        env = MacroEnv(config=EconomyConfig(max_steps=5), reward_config=RewardConfig())
        check_env(env, skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
