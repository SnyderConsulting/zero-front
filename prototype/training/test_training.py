import argparse
import unittest
import numpy as np

from training.train import ValidationPlateau, known_training_seeds, seed_list, validate_dataset


class TrainingConfigurationTests(unittest.TestCase):
    def test_seed_ranges_are_bounded_and_exclusive(self):
        self.assertEqual(seed_list("10:13,20"), [10, 11, 12, 20])
        self.assertEqual(seed_list("10,10,11"), [10, 11])
        for value in ("-1", "4294967296", "10:10", "0:129"):
            with self.assertRaises(argparse.ArgumentTypeError):
                seed_list(value)

    def test_warmstart_lineage_preserves_all_prior_generations(self):
        self.assertEqual(known_training_seeds({"trainSeeds": [10, 11], "priorTrainingSeeds": [2, 3, 10]}), {2, 3, 10, 11})
        self.assertEqual(known_training_seeds({}), set())
        for metadata in ({"trainSeeds": [True]}, {"priorTrainingSeeds": ["2"]}, {"trainSeeds": [-1]}, []):
            with self.assertRaises(ValueError):
                known_training_seeds(metadata)

    def test_validation_patience_requires_material_improvement(self):
        plateau = ValidationPlateau(1.0, patience=2, min_delta=.001)
        self.assertFalse(plateau.update(.9))
        self.assertFalse(plateau.update(.8995))
        self.assertFalse(plateau.update(.898))
        self.assertFalse(plateau.update(.899))
        self.assertTrue(plateau.update(.8985))
        disabled = ValidationPlateau(1.0, patience=0, min_delta=.001)
        for _ in range(20):
            self.assertFalse(disabled.update(1.1))
        with self.assertRaises(ValueError):
            disabled.update(float('nan'))

    def test_corrupt_or_empty_demonstrations_are_rejected(self):
        pixels = np.zeros((2, 36, 64, 3), dtype=np.uint8)
        scalars = np.zeros((2, 6), dtype=np.float32)
        labels = np.zeros((2, 6), dtype=np.int64)
        validate_dataset((pixels, scalars, labels))
        with self.assertRaises(ValueError):
            validate_dataset((pixels[:0], scalars[:0], labels[:0]))
        with self.assertRaises(ValueError):
            validate_dataset((pixels, scalars, labels.astype(np.float32)))
        labels[0, 2] = 100
        with self.assertRaises(ValueError):
            validate_dataset((pixels, scalars, labels))


if __name__ == "__main__":
    unittest.main()
