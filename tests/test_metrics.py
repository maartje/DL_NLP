import unittest
from src.reporting.metrics import calculate_accuracy, calculate_confusion_matrix
import numpy as np

class TestMetrics(unittest.TestCase):

    def test_confusion_matrix(self):
        probs_1 = [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3],
            [0.4, 0.3, 0.3]
        ]
        probs_2 = [
            [0.2, 0.2, 0.6],
            [0.3, 0.2, 0.5],
            [0.2, 0.1, 0.7],
            [0.9, 0.1, 0.0]
        ]
        probs = np.stack([probs_1, probs_2])

        targets = np.array([
            [1, 1, 1, 1],
            [2, 2, 2, 2]
        ])

        counts_confusion_matrix = calculate_confusion_matrix(probs, targets, counts=True)[0]
        percentage_confusion_matrix = calculate_confusion_matrix(probs, targets, counts=False)[0]

        self.assertTrue(np.array_equal(counts_confusion_matrix, [[1, 2, 1], [1, 0, 3]]))
        self.assertTrue(np.array_equal(percentage_confusion_matrix, [[0.25, 0.5, 0.25], [0.25, 0.0, 0.75]]))

    def test_accuracy(self):
        probs_1 = [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3],
            [0.4, 0.3, 0.3]
        ]
        probs_2 = [
            [0.2, 0.2, 0.6],
            [0.3, 0.2, 0.5],
            [0.2, 0.1, 0.7],
            [0.9, 0.1, 0.0]    
        ]
        probs = np.stack([probs_1, probs_2])

        targets = np.array([
            [1,1,1,1],
            [2,2,2,2]
        ])

        avg_acc = calculate_accuracy(probs, targets, t_axis=None)
        seq_acc = calculate_accuracy(probs, targets, t_axis=0)
        instances_acc = calculate_accuracy(probs, targets, t_axis=1)

        self.assertEqual(avg_acc, seq_acc.mean())
        self.assertEqual(avg_acc, instances_acc.mean())
        self.assertEqual(avg_acc, 5/8)
        self.assertTrue(np.array_equal(seq_acc, [1/2, 2/2, 2/2, 0]))
        self.assertTrue(np.array_equal(instances_acc, [2/4, 3/4]))

    def test_accuracy_padding(self):
        probs_1 = [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.4, 0.3, 0.3],
            [0.2, 0.5, 0.3],
        ]
        probs_2 = [
            [0.2, 0.2, 0.6],
            [0.3, 0.2, 0.5],
            [0.2, 0.1, 0.7],
            [0.0, 0.9, 0.1]    
        ]
        probs = np.stack([probs_1, probs_2])

        targets = np.array([
            [1,1,1,1],
            [2,2,2,0]
        ])

        avg_acc = calculate_accuracy(probs, targets, t_axis=None)
        seq_acc = calculate_accuracy(probs, targets, t_axis=0)
        instances_acc = calculate_accuracy(probs, targets, t_axis=1)

        self.assertEqual(avg_acc, 5/7)
        self.assertTrue(np.array_equal(seq_acc, [1/2, 2/2, 1/2, 1.]))
        self.assertTrue(np.array_equal(instances_acc, [2/4, 3/3]))
