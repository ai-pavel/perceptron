"""Tests for Sequential model."""

import tempfile
import numpy as np
from nn import Dense, ReLU, Softmax, Sequential, CrossEntropy, SGD


class TestSequential:
    def _make_model(self):
        return Sequential([Dense(4, 3), ReLU(), Dense(3, 2), Softmax()])

    def test_forward_shape(self):
        model = self._make_model()
        out = model.predict(np.random.randn(5, 4))
        assert out.shape == (5, 2)

    def test_predict_sums_to_one(self):
        model = self._make_model()
        out = model.predict(np.random.randn(5, 4))
        np.testing.assert_allclose(np.sum(out, axis=1), np.ones(5), atol=1e-7)

    def test_fit_decreases_loss(self):
        np.random.seed(0)
        model = Sequential([Dense(4, 8), ReLU(), Dense(8, 2), Softmax()])
        x = np.random.randn(100, 4)
        y = np.zeros((100, 2))
        y[np.arange(100), np.random.randint(0, 2, 100)] = 1.0

        loss_fn = CrossEntropy()
        opt = SGD(lr=0.05)
        history = model.fit(x, y, loss_fn, opt, epochs=20, batch_size=32, verbose=False)
        assert history["loss"][-1] < history["loss"][0]

    def test_save_load_roundtrip(self):
        np.random.seed(42)
        model = self._make_model()
        x = np.random.randn(3, 4)
        pred_before = model.predict(x)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            model.save(f.name)
            loaded = Sequential.load(f.name)
            pred_after = loaded.predict(x)

        np.testing.assert_array_equal(pred_before, pred_after)
