"""Run the distributed Lasso workflow on fully synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distributed_lasso import (  # noqa: E402
    Node,
    federated_train,
    fit_lasso,
    select_regularization,
    support_confusion,
)


def make_node_data(
    rng: np.random.Generator,
    sample_count: int,
    true_beta: np.ndarray,
    feature_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_count = true_beta.size
    X = rng.normal(feature_shift, 1.0, size=(sample_count, feature_count))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = X @ true_beta + rng.normal(0.0, 1.2, size=sample_count)
    split = int(sample_count * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


def main() -> None:
    rng = np.random.default_rng(42)
    feature_count = 120
    true_beta = np.zeros(feature_count)
    active = rng.choice(feature_count, size=12, replace=False)
    true_beta[active] = rng.normal(0.0, 3.0, size=active.size)

    node_sizes = [200, 300, 500]
    shifts = [-0.15, 0.0, 0.15]
    candidate_lambdas = [0.03, 0.06, 0.1, 0.16, 0.25]

    nodes: list[Node] = []
    train_parts: list[tuple[np.ndarray, np.ndarray]] = []
    validation_parts: list[tuple[np.ndarray, np.ndarray]] = []

    for index, (size, shift) in enumerate(zip(node_sizes, shifts), start=1):
        X_train, y_train, X_val, y_val = make_node_data(
            rng, size, true_beta, shift
        )
        best_lambda, scores = select_regularization(
            X_train,
            y_train,
            X_val,
            y_val,
            candidate_lambdas,
        )
        node_id = f"node-{index}"
        nodes.append(Node(node_id, X_train, y_train, best_lambda))
        train_parts.append((X_train, y_train))
        validation_parts.append((X_val, y_val))
        print(f"{node_id}: lambda={best_lambda:.3f}, validation={scores[best_lambda]:.4f}")

    pooled_X = np.vstack([part[0] for part in train_parts])
    pooled_y = np.concatenate([part[1] for part in train_parts])
    pooled_X_val = np.vstack([part[0] for part in validation_parts])
    pooled_y_val = np.concatenate([part[1] for part in validation_parts])
    pooled_lambda, _ = select_regularization(
        pooled_X,
        pooled_y,
        pooled_X_val,
        pooled_y_val,
        candidate_lambdas,
    )
    pooled_beta = fit_lasso(pooled_X, pooled_y, pooled_lambda)

    print("\nE | rounds | converged | active | TP | FP | FN | TN")
    print("-" * 55)
    for local_epochs in (1, 5, 10):
        result = federated_train(nodes, local_epochs=local_epochs)
        metrics = support_confusion(result.beta, pooled_beta)
        print(
            f"{local_epochs:>1} | {result.rounds:>6} | {str(result.converged):>9} | "
            f"{metrics['active']:>6} | {metrics['tp']:>2} | {metrics['fp']:>2} | "
            f"{metrics['fn']:>2} | {metrics['tn']:>2}"
        )


if __name__ == "__main__":
    main()

