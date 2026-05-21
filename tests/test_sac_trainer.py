import os
import sys
import unittest
from pathlib import Path

from stable_baselines3 import SAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rl import (
    SACTrainerConfig,
    build_callbacks,
    build_model,
    build_training_env,
    save_run_metadata,
)


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

    def test_build_callbacks_returns_auxiliary_eval_env(self):
        trainer_config = SACTrainerConfig(
            total_timesteps=1,
            eval_freq=10,
            n_eval_episodes=1,
            checkpoint_freq=10,
            metrics_path="outputs/logs/test_episode_metrics.jsonl",
            checkpoint_dir="outputs/models/test_checkpoints",
            best_model_dir="outputs/models/test_best",
            eval_log_dir="outputs/logs/test_eval",
        )
        callbacks, auxiliary_envs = build_callbacks(trainer_config)
        try:
            self.assertGreaterEqual(len(callbacks.callbacks), 3)
            self.assertEqual(len(auxiliary_envs), 1)
        finally:
            for env in auxiliary_envs:
                env.close()

    def test_save_run_metadata_writes_json(self):
        trainer_config = SACTrainerConfig(model_path="outputs/models/test_sac_model")
        metadata_path = save_run_metadata(trainer_config)

        self.assertTrue(Path(metadata_path).exists())


if __name__ == "__main__":
    unittest.main()
