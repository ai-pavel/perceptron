"""Neural network layer implementations."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """Base class for all layers."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.training: bool = True

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        ...

    def get_params(self) -> dict[str, np.ndarray]:
        return self.params

    def get_grads(self) -> dict[str, np.ndarray]:
        return self.grads


class Dense(Layer):
    """Fully connected layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # He initialization
        scale = np.sqrt(2.0 / in_features)
        self.params["W"] = np.random.randn(in_features, out_features) * scale
        self.params["b"] = np.zeros((1, out_features))
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._input is not None
        self.grads["W"] = self._input.T @ grad
        self.grads["b"] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params["W"].T


class ReLU(Layer):
    """Rectified Linear Unit activation."""

    def __init__(self) -> None:
        super().__init__()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0).astype(x.dtype)
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._mask is not None
        return grad * self._mask


class Sigmoid(Layer):
    """Sigmoid activation."""

    def __init__(self) -> None:
        super().__init__()
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._out is not None
        return grad * self._out * (1.0 - self._out)


class Softmax(Layer):
    """Softmax activation."""

    def __init__(self) -> None:
        super().__init__()
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self._out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # When used with CrossEntropy, the combined gradient is passed
        # directly, so we just pass through.
        return grad


class Dropout(Layer):
    """Dropout regularization layer."""

    def __init__(self, rate: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = rate
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x
        self._mask = (np.random.rand(*x.shape) > self.rate).astype(x.dtype)
        return x * self._mask / (1.0 - self.rate)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad
        assert self._mask is not None
        return grad * self._mask / (1.0 - self.rate)
