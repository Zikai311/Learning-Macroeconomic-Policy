import os
import sys
import unittest

from stable_baselines3 import SAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rl import SACTrainerConfig, build_model, build_training_env


class SACTrainerScaffoldTests(unittest.TestCase):
    def test_build_training_env_can_normalize_actions(self):
        env = build_training_env(normalize_actions=True, seed=123)
        try:
            self.assertEqual(env.action_space.shape, (3,))
            self.assertTrue((env.action_space.low == -1.0).all())
            self.assertTrue((env.action_space.high == 1.0).all())
        finally:
            env.close()

    def test_build_model_returns_sac_instance(self):
        env = build_training_env(normalize_actions=True, seed=123)
        try:
            model = build_model(env, SACTrainerConfig(total_timesteps=1))
            self.assertIsInstance(model, SAC)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
