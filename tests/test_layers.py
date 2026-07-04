"""Tests for layer implementations."""

import numpy as np
import pytest
from nn.layers import Dense, ReLU, Sigmoid, Softmax, Dropout


class TestDense:
    def test_forward_shape(self):
        layer = Dense(4, 3)
        x = np.random.randn(5, 4)
        out = layer.forward(x)
        assert out.shape == (5, 3)

    def test_backward_shape(self):
        layer = Dense(4, 3)
        x = np.random.randn(5, 4)
        out = layer.forward(x)
        grad = np.ones_like(out)
        dx = layer.backward(grad)
        assert dx.shape == (5, 4)
        assert layer.grads["W"].shape == (4, 3)
        assert layer.grads["b"].shape == (1, 3)

    def test_gradient_numerically(self):
        np.random.seed(42)
        layer = Dense(3, 2)
        x = np.random.randn(4, 3)
        out = layer.forward(x)
        grad = np.ones_like(out)
        layer.backward(grad)

        eps = 1e-5
        for i in range(3):
            for j in range(2):
                layer.params["W"][i, j] += eps
                out_plus = layer.forward(x)
                loss_plus = np.sum(out_plus)

                layer.params["W"][i, j] -= 2 * eps
                out_minus = layer.forward(x)
                loss_minus = np.sum(out_minus)

                layer.params["W"][i, j] += eps  # restore
                num_grad = (loss_plus - loss_minus) / (2 * eps)
                np.testing.assert_allclose(
                    layer.grads["W"][i, j], num_grad, atol=1e-5
                )


class TestReLU:
    def test_forward(self):
        layer = ReLU()
        x = np.array([[-1, 2], [3, -4]])
        out = layer.forward(x)
        np.testing.assert_array_equal(out, [[0, 2], [3, 0]])

    def test_backward(self):
        layer = ReLU()
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        layer.forward(x)
        grad = np.ones_like(x)
        dx = layer.backward(grad)
        np.testing.assert_array_equal(dx, [[0, 1], [1, 0]])


class TestSigmoid:
    def test_output_range(self):
        layer = Sigmoid()
        x = np.random.randn(10, 5)
        out = layer.forward(x)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_known_value(self):
        layer = Sigmoid()
        out = layer.forward(np.array([[0.0]]))
        np.testing.assert_allclose(out, [[0.5]])


class TestSoftmax:
    def test_sums_to_one(self):
        layer = Softmax()
        x = np.random.randn(5, 4)
        out = layer.forward(x)
        np.testing.assert_allclose(np.sum(out, axis=1), np.ones(5), atol=1e-7)

    def test_all_positive(self):
        layer = Softmax()
        x = np.array([[-100, 0, 100]])
        out = layer.forward(x)
        assert np.all(out >= 0)


class TestDropout:
    def test_training_scaling(self):
        np.random.seed(0)
        layer = Dropout(rate=0.5)
        layer.training = True
        x = np.ones((1000, 100))
        out = layer.forward(x)
        # Mean should be approximately 1.0 due to inverted dropout scaling
        np.testing.assert_allclose(np.mean(out), 1.0, atol=0.1)

    def test_inference_passthrough(self):
        layer = Dropout(rate=0.5)
        layer.training = False
        x = np.ones((10, 5))
        out = layer.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            Dropout(rate=1.0)
