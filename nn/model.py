"""Sequential model implementation."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from .layers import Layer, Dense, ReLU, Sigmoid, Softmax, Dropout
from .losses import Loss
from .optimizers import Optimizer


class Sequential:
    """Sequential neural network model."""

    def __init__(self, layers: list[Layer] | None = None) -> None:
        self.layers: list[Layer] = layers or []

    def add(self, layer: Layer) -> "Sequential":
        self.layers.append(layer)
        return self

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            layer.training = training
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, training=False)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        loss_fn: Loss,
        optimizer: Optimizer,
        epochs: int = 10,
        batch_size: int = 32,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {"loss": [], "val_loss": [], "val_acc": []}
        n = x_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle
            perm = np.random.permutation(n)
            x_shuffled = x_train[perm]
            y_shuffled = y_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = x_shuffled[start:end]
                yb = y_shuffled[start:end]

                # Forward
                out = self.forward(xb, training=True)
                batch_loss = loss_fn.forward(out, yb)
                epoch_loss += batch_loss
                n_batches += 1

                # Backward
                grad = loss_fn.backward(out, yb)
                self.backward(grad)

                # Update
                optimizer.step(self.layers)

            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)

            # Validation
            msg = f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}"
            if x_val is not None and y_val is not None:
                val_out = self.predict(x_val)
                val_loss = loss_fn.forward(val_out, y_val)
                history["val_loss"].append(val_loss)

                val_preds = np.argmax(val_out, axis=1)
                val_true = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
                val_acc = float(np.mean(val_preds == val_true))
                history["val_acc"].append(val_acc)
                msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"

            if verbose:
                print(msg)

        return history

    def save(self, path: str | Path) -> None:
        """Save model parameters to a .npz file with metadata."""
        path = Path(path)
        save_dict: dict[str, np.ndarray] = {}
        meta: list[dict] = []

        for i, layer in enumerate(self.layers):
            layer_info: dict = {"type": type(layer).__name__}
            if isinstance(layer, Dense):
                layer_info["in_features"] = int(layer.params["W"].shape[0])
                layer_info["out_features"] = int(layer.params["W"].shape[1])
                save_dict[f"layer_{i}_W"] = layer.params["W"]
                save_dict[f"layer_{i}_b"] = layer.params["b"]
            elif isinstance(layer, Dropout):
                layer_info["rate"] = layer.rate
            meta.append(layer_info)

        # Save metadata as a JSON string stored in an array
        save_dict["_meta"] = np.array(json.dumps(meta))
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str | Path) -> "Sequential":
        """Load model from a .npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["_meta"]))

        layer_classes = {
            "Dense": Dense,
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "Softmax": Softmax,
            "Dropout": Dropout,
        }

        layers: list[Layer] = []
        for i, info in enumerate(meta):
            ltype = info["type"]
            if ltype == "Dense":
                layer = Dense(info["in_features"], info["out_features"])
                layer.params["W"] = data[f"layer_{i}_W"]
                layer.params["b"] = data[f"layer_{i}_b"]
            elif ltype == "Dropout":
                layer = Dropout(rate=info["rate"])
            else:
                layer = layer_classes[ltype]()
            layers.append(layer)

        return cls(layers)
