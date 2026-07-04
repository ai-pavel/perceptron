"""Tests for optimizers."""

import numpy as np
from nn.layers import Dense
from nn.optimizers import SGD, Adam


class TestSGD:
    def test_step_updates_params(self):
        layer = Dense(3, 2)
        w_before = layer.params["W"].copy()
        layer.grads["W"] = np.ones_like(layer.params["W"])
        layer.grads["b"] = np.ones_like(layer.params["b"])
        opt = SGD(lr=0.1)
        opt.step([layer])
        assert not np.array_equal(layer.params["W"], w_before)

    def test_momentum(self):
        layer = Dense(3, 2)
        layer.grads["W"] = np.ones_like(layer.params["W"])
        layer.grads["b"] = np.zeros_like(layer.params["b"])
        opt = SGD(lr=0.1, momentum=0.9)
        w0 = layer.params["W"].copy()
        opt.step([layer])
        delta1 = layer.params["W"] - w0

        w1 = layer.params["W"].copy()
        opt.step([layer])
        delta2 = layer.params["W"] - w1
        # With momentum the second step should be larger
        assert np.linalg.norm(delta2) > np.linalg.norm(delta1)


class TestAdam:
    def test_step_updates_params(self):
        layer = Dense(3, 2)
        w_before = layer.params["W"].copy()
        layer.grads["W"] = np.ones_like(layer.params["W"])
        layer.grads["b"] = np.ones_like(layer.params["b"])
        opt = Adam(lr=0.01)
        opt.step([layer])
        assert not np.array_equal(layer.params["W"], w_before)
