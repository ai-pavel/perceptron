"""Tests for loss functions."""

import numpy as np
from nn.losses import MSE, CrossEntropy


class TestMSE:
    def test_zero_loss(self):
        loss = MSE()
        pred = np.array([[1.0, 2.0]])
        assert loss(pred, pred) == 0.0

    def test_known_value(self):
        loss = MSE()
        pred = np.array([[1.0, 0.0]])
        target = np.array([[0.0, 1.0]])
        # ((1-0)^2 + (0-1)^2) / 2 = 1.0
        np.testing.assert_allclose(loss(pred, target), 1.0)

    def test_backward_shape(self):
        loss = MSE()
        pred = np.random.randn(5, 3)
        target = np.random.randn(5, 3)
        grad = loss.backward(pred, target)
        assert grad.shape == pred.shape


class TestCrossEntropy:
    def test_perfect_prediction(self):
        loss = CrossEntropy()
        pred = np.array([[0.99, 0.01]])
        target = np.array([[1.0, 0.0]])
        assert loss(pred, target) < 0.02

    def test_backward_shape(self):
        loss = CrossEntropy()
        pred = np.array([[0.7, 0.2, 0.1]])
        target = np.array([[1.0, 0.0, 0.0]])
        grad = loss.backward(pred, target)
        assert grad.shape == pred.shape
