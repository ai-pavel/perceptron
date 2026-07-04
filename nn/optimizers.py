"""Optimizer implementations."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from .layers import Layer


class Optimizer(ABC):
    """Base class for optimizers."""

    @abstractmethod
    def step(self, layers: list[Layer]) -> None:
        ...


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self._velocities: dict[int, dict[str, np.ndarray]] = {}

    def step(self, layers: list[Layer]) -> None:
        for i, layer in enumerate(layers):
            if not layer.params:
                continue
            if i not in self._velocities:
                self._velocities[i] = {
                    k: np.zeros_like(v) for k, v in layer.params.items()
                }
            for key in layer.params:
                v = self._velocities[i][key]
                g = layer.grads[key]
                v[:] = self.momentum * v - self.lr * g
                layer.params[key] += v


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: dict[int, dict[str, np.ndarray]] = {}
        self._v: dict[int, dict[str, np.ndarray]] = {}

    def step(self, layers: list[Layer]) -> None:
        self.t += 1
        for i, layer in enumerate(layers):
            if not layer.params:
                continue
            if i not in self._m:
                self._m[i] = {
                    k: np.zeros_like(v) for k, v in layer.params.items()
                }
                self._v[i] = {
                    k: np.zeros_like(v) for k, v in layer.params.items()
                }
            for key in layer.params:
                g = layer.grads[key]
                self._m[i][key] = self.beta1 * self._m[i][key] + (1 - self.beta1) * g
                self._v[i][key] = self.beta2 * self._v[i][key] + (1 - self.beta2) * g ** 2
                m_hat = self._m[i][key] / (1 - self.beta1 ** self.t)
                v_hat = self._v[i][key] / (1 - self.beta2 ** self.t)
                layer.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
