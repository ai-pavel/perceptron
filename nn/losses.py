"""Loss function implementations."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """Base class for loss functions."""

    @abstractmethod
    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        ...

    @abstractmethod
    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, predicted: np.ndarray, target: np.ndarray) -> float:
        return self.forward(predicted, target)


class MSE(Loss):
    """Mean Squared Error loss."""

    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((predicted - target) ** 2))

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        n = predicted.shape[0]
        return 2.0 * (predicted - target) / n


class CrossEntropy(Loss):
    """Cross-entropy loss (expects softmax output and one-hot targets)."""

    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        eps = 1e-12
        clipped = np.clip(predicted, eps, 1.0 - eps)
        return float(-np.mean(np.sum(target * np.log(clipped), axis=1)))

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Combined softmax + cross-entropy gradient
        n = predicted.shape[0]
        return (predicted - target) / n
