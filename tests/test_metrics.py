import unittest
from src.reporting.metrics import calculate_accuracy
import numpy as np

class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        probs_1 = [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.2, 0.4, 0.4],
            [0.4, 0.3, 0.3]
        ]
        probs_2 = [
            [0.2, 0.6, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.1, 0.7],
            [0.9, 0.1, 0.0]    
        ]
        probs = np.stack([probs_1, probs_2])

        targets = np.array([
            [1,1,1,1],
            [1,0,2,2]
        ])

        avg_acc = calculate_accuracy(probs, targets, t_axis=None)
        seq_acc = calculate_accuracy(probs, targets, t_axis=0)
        instances_acc = calculate_accuracy(probs, targets, t_axis=1)

        self.assertEqual(avg_acc, seq_acc.mean())
        self.assertEqual(avg_acc, instances_acc.mean())
        self.assertEqual(avg_acc, 5/8)
        self.assertTrue(np.array_equal(seq_acc, [1/2, 2/2, 2/2, 0]))
        self.assertTrue(np.array_equal(instances_acc, [2/4, 3/4]))
