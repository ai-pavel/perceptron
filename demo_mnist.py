#!/usr/bin/env python3
"""Demo: Train a neural network on MNIST and achieve >95% accuracy."""

import numpy as np
from nn import Dense, ReLU, Softmax, Dropout, Sequential, CrossEntropy, Adam


def load_mnist():
    """Load MNIST dataset via sklearn."""
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(int)
    return X, y


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    oh = np.zeros((y.shape[0], num_classes))
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def main():
    print("Loading MNIST...")
    X, y = load_mnist()

    # Split: 60k train, 10k test
    x_train, x_test = X[:60000], X[60000:]
    y_train_raw, y_test_raw = y[:60000], y[60000:]
    y_train = one_hot(y_train_raw)
    y_test = one_hot(y_test_raw)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = Sequential([
        Dense(784, 256),
        ReLU(),
        Dropout(0.2),
        Dense(256, 128),
        ReLU(),
        Dropout(0.2),
        Dense(128, 10),
        Softmax(),
    ])

    loss_fn = CrossEntropy()
    optimizer = Adam(lr=0.001)

    history = model.fit(
        x_train, y_train,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=10,
        batch_size=128,
        x_val=x_test,
        y_val=y_test,
        verbose=True,
    )

    # Final evaluation
    preds = model.predict(x_test)
    accuracy = np.mean(np.argmax(preds, axis=1) == y_test_raw)
    print(f"\nFinal test accuracy: {accuracy:.4f}")

    if accuracy > 0.95:
        print("Target accuracy (>95%) achieved!")
    else:
        print("Target accuracy not yet reached — consider more epochs or tuning.")

    # Save and reload
    model.save("mnist_model.npz")
    loaded = Sequential.load("mnist_model.npz")
    preds2 = loaded.predict(x_test)
    acc2 = np.mean(np.argmax(preds2, axis=1) == y_test_raw)
    print(f"Loaded model accuracy: {acc2:.4f}")


if __name__ == "__main__":
    main()
