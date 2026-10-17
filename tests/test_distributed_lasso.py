import unittest

import numpy as np

from src.distributed_lasso import (
    Node,
    federated_train,
    fit_lasso,
    soft_threshold,
    support_confusion,
)


class DistributedLassoTests(unittest.TestCase):
    def test_soft_threshold(self) -> None:
        values = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = soft_threshold(values, 1.0)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0, 0.0, 1.0])

    def test_lasso_recovers_simple_signal(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(300, 5))
        expected = np.array([2.0, 0.0, -1.5, 0.0, 0.0])
        y = X @ expected + rng.normal(scale=0.05, size=300)
        estimate = fit_lasso(X, y, 0.05)
        self.assertGreater(abs(estimate[0]), 1.0)
        self.assertGreater(abs(estimate[2]), 0.8)
        self.assertLess(np.max(np.abs(estimate[[1, 3, 4]])), 0.1)

    def test_federated_training_converges(self) -> None:
        rng = np.random.default_rng(11)
        beta = np.array([1.5, 0.0, -1.0, 0.0])
        nodes = []
        for index, size in enumerate((80, 120, 200), start=1):
            X = rng.normal(size=(size, 4))
            y = X @ beta + rng.normal(scale=0.2, size=size)
            nodes.append(Node(str(index), X, y, 0.05))
        result = federated_train(nodes, local_epochs=1, max_rounds=200)
        self.assertTrue(result.converged)
        self.assertGreater(abs(result.beta[0]), 1.0)
        self.assertGreater(abs(result.beta[2]), 0.5)

    def test_support_confusion(self) -> None:
        metrics = support_confusion(
            np.array([1.0, 1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 1.0, 0.0]),
        )
        self.assertEqual(metrics, {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "active": 2})


if __name__ == "__main__":
    unittest.main()

