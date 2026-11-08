import unittest
import numpy as np
from training.ppo import advantages, seed_ranges

class PPOTests(unittest.TestCase):
    def test_gae_does_not_bootstrap_from_autoreset_observation(self):
        rewards = np.array([[2.], [3.]], dtype=np.float32)
        values = np.array([[1.], [100.]], dtype=np.float32)
        dones = np.array([[1.], [0.]], dtype=np.float32)
        advantage, returns = advantages(rewards, values, dones, np.array([200.]), gamma=.5, lam=1)
        np.testing.assert_allclose(advantage[:, 0], [1, 3])
        np.testing.assert_allclose(returns[:, 0], [2, 103])

    def test_gae_bootstraps_unfinished_rollout(self):
        rewards = np.array([[2.], [3.]], dtype=np.float32)
        values = np.array([[1.], [4.]], dtype=np.float32)
        advantage, returns = advantages(rewards, values, np.zeros_like(rewards), np.array([6.]), gamma=.5, lam=1)
        np.testing.assert_allclose(returns[:, 0], [5, 6])
        np.testing.assert_allclose(advantage[:, 0], [4, 2])

    def test_seed_lineage_compacts_without_forgetting_previous_ranges(self):
        self.assertEqual(seed_ranges([1, 2, 3, 12], [[3, 6], [10, 12]]), [[1, 6], [10, 13]])
        self.assertEqual(seed_ranges(list(range(200))), [[0, 200]])

if __name__ == "__main__":unittest.main()
