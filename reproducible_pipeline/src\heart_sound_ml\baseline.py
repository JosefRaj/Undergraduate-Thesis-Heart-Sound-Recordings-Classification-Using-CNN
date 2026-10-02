from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer has not been fitted")
        return (x - self.mean_) / self.scale_


class LogisticBaseline:
    """Small NumPy logistic-regression baseline for reproducible comparisons."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 1500, l2: float = 1e-3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.scaler = Standardizer()
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xs = self.scaler.fit(x).transform(x)
        self.weights_ = np.zeros(xs.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(xs @ self.weights_ + self.bias_)
            error = p - y
            grad_w = xs.T @ error / len(xs) + self.l2 * self.weights_
            grad_b = float(np.mean(error))
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted")
        xs = self.scaler.transform(np.asarray(x, dtype=np.float64))
        return self._sigmoid(xs @ self.weights_ + self.bias_)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)
