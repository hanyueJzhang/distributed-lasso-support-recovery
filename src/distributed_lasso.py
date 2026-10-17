"""Synchronous distributed Lasso using local coordinate-descent updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


Array = np.ndarray


def soft_threshold(value: Array | float, threshold: float) -> Array | float:
    """Apply the Lasso soft-thresholding operator."""
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def _feature_norms(X: Array) -> Array:
    norms = np.mean(np.square(X), axis=0)
    if np.any(norms <= 0):
        raise ValueError("Every feature must have positive variance or magnitude.")
    return norms


def coordinate_descent_epoch(
    X: Array,
    y: Array,
    beta: Array,
    regularization: float,
    feature_norms: Array,
    residual: Array,
) -> tuple[Array, Array]:
    """Run one sequential coordinate-descent pass in place."""
    sample_count = X.shape[0]
    for coordinate in range(X.shape[1]):
        previous = beta[coordinate]
        correlation = (
            X[:, coordinate] @ residual / sample_count
            + feature_norms[coordinate] * previous
        )
        beta[coordinate] = (
            soft_threshold(correlation, regularization)
            / feature_norms[coordinate]
        )
        residual += X[:, coordinate] * (previous - beta[coordinate])
    return beta, residual


def fit_lasso(
    X: Array,
    y: Array,
    regularization: float,
    *,
    initial_beta: Array | None = None,
    max_epochs: int = 500,
    tolerance: float = 1e-6,
) -> Array:
    """Fit a Lasso model to convergence with coordinate descent."""
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2D and y must be a matching 1D vector.")
    if regularization < 0:
        raise ValueError("regularization must be non-negative.")

    beta = (
        np.zeros(X.shape[1], dtype=float)
        if initial_beta is None
        else np.asarray(initial_beta, dtype=float).copy()
    )
    if beta.shape != (X.shape[1],):
        raise ValueError("initial_beta has the wrong shape.")

    feature_norms = _feature_norms(X)
    residual = y - X @ beta
    for _ in range(max_epochs):
        previous = beta.copy()
        beta, residual = coordinate_descent_epoch(
            X, y, beta, regularization, feature_norms, residual
        )
        if np.linalg.norm(beta - previous) < tolerance:
            break
    return beta


def mean_squared_error(y_true: Array, y_pred: Array) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def select_regularization(
    X_train: Array,
    y_train: Array,
    X_validation: Array,
    y_validation: Array,
    candidates: Iterable[float],
) -> tuple[float, dict[float, float]]:
    """Choose the candidate with the lowest validation MSE."""
    scores: dict[float, float] = {}
    for value in candidates:
        regularization = float(value)
        beta = fit_lasso(X_train, y_train, regularization)
        scores[regularization] = mean_squared_error(
            y_validation, X_validation @ beta
        )
    if not scores:
        raise ValueError("At least one regularization candidate is required.")
    best = min(scores, key=scores.get)
    return best, scores


@dataclass
class Node:
    """A data owner whose observations stay inside the local object."""

    node_id: str
    X: Array
    y: Array
    regularization: float

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.X.ndim != 2 or self.y.shape != (self.X.shape[0],):
            raise ValueError("Node data dimensions do not match.")
        self.feature_norms = _feature_norms(self.X)

    @property
    def sample_count(self) -> int:
        return self.X.shape[0]

    @property
    def feature_count(self) -> int:
        return self.X.shape[1]

    def local_update(self, global_beta: Array, local_epochs: int) -> tuple[Array, float]:
        beta = np.asarray(global_beta, dtype=float).copy()
        residual = self.y - self.X @ beta
        for _ in range(local_epochs):
            beta, residual = coordinate_descent_epoch(
                self.X,
                self.y,
                beta,
                self.regularization,
                self.feature_norms,
                residual,
            )
        loss = (
            0.5 * np.mean(np.square(residual))
            + self.regularization * np.sum(np.abs(beta))
        )
        return beta, float(loss)


@dataclass(frozen=True)
class FederatedResult:
    beta: Array
    loss_history: tuple[float, ...]
    rounds: int
    converged: bool


def federated_train(
    nodes: Sequence[Node],
    *,
    local_epochs: int,
    weights: Mapping[str, float] | None = None,
    tolerance: float = 1e-5,
    max_rounds: int = 500,
) -> FederatedResult:
    """Train a synchronous sample-weighted consensus model."""
    if not nodes:
        raise ValueError("At least one node is required.")
    if local_epochs < 1:
        raise ValueError("local_epochs must be at least one.")
    feature_count = nodes[0].feature_count
    if any(node.feature_count != feature_count for node in nodes):
        raise ValueError("All nodes must use the same feature dimension.")

    if weights is None:
        total = sum(node.sample_count for node in nodes)
        normalized = {node.node_id: node.sample_count / total for node in nodes}
    else:
        missing = {node.node_id for node in nodes} - set(weights)
        if missing:
            raise ValueError(f"Missing weights for nodes: {sorted(missing)}")
        weight_sum = sum(float(weights[node.node_id]) for node in nodes)
        if weight_sum <= 0:
            raise ValueError("Weights must have a positive sum.")
        normalized = {
            node.node_id: float(weights[node.node_id]) / weight_sum for node in nodes
        }

    beta = np.zeros(feature_count, dtype=float)
    history: list[float] = []
    for round_number in range(1, max_rounds + 1):
        updates = [node.local_update(beta, local_epochs) for node in nodes]
        next_beta = sum(
            normalized[node.node_id] * update[0]
            for node, update in zip(nodes, updates)
        )
        history.append(
            sum(
                normalized[node.node_id] * update[1]
                for node, update in zip(nodes, updates)
            )
        )
        change = np.linalg.norm(next_beta - beta)
        beta = next_beta
        if change < tolerance:
            return FederatedResult(beta, tuple(history), round_number, True)

    return FederatedResult(beta, tuple(history), max_rounds, False)


def active_support(beta: Array, threshold: float = 1e-6) -> Array:
    return np.abs(beta) > threshold


def support_confusion(
    estimate: Array, reference: Array, threshold: float = 1e-6
) -> dict[str, int]:
    """Compare active coefficients with a reference active set."""
    estimated = active_support(estimate, threshold)
    expected = active_support(reference, threshold)
    if estimated.shape != expected.shape:
        raise ValueError("estimate and reference must have the same shape.")
    return {
        "tp": int(np.sum(estimated & expected)),
        "fp": int(np.sum(estimated & ~expected)),
        "fn": int(np.sum(~estimated & expected)),
        "tn": int(np.sum(~estimated & ~expected)),
        "active": int(np.sum(estimated)),
    }

