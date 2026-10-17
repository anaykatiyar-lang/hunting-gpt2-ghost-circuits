import unittest

import numpy as np
try:
    import torch
    from src.metrics import (
        doubles_advantage,
        doubles_score,
        own_answer_contrast,
        verify_dla_sum,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class MetricTests(unittest.TestCase):
    def test_doubles_score_uses_mean_of_neighbor_logits(self):
        logits = torch.tensor([[[0.0, 2.0, 8.0, 6.0]]])
        self.assertEqual(doubles_score(logits, target_id=2, neighbor_ids=(1, 3)), 4.0)

    def test_doubles_advantage_averages_controls(self):
        self.assertEqual(doubles_advantage(5.0, [1.0, 3.0]), 3.0)

    def test_doubles_advantage_rejects_missing_controls(self):
        with self.assertRaises(ValueError):
            doubles_advantage(5.0, [])

    def test_dla_reconstruction_includes_unembedding_bias(self):
        passes, residual = verify_dla_sum([0.4, 0.3], 0.2, 0.1, 0.8)
        self.assertTrue(passes)
        self.assertAlmostEqual(residual, 0.0)

    def test_own_answer_contrast_and_missing_class(self):
        logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(own_answer_contrast(logits, labels, [0, 1]), 0.5)

        with self.assertRaisesRegex(ValueError, "No prompts have answer label 1"):
            own_answer_contrast(np.array([[1.0, 0.0], [2.0, 0.0]]), np.array([0, 2]), [0, 1])

    def test_own_answer_contrast_rejects_nonfinite_logits(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            own_answer_contrast(np.array([[np.nan, 0.0], [0.0, 1.0]]), np.array([0, 1]), [0, 1])


if __name__ == "__main__":
    unittest.main()
