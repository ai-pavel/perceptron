# perceptron

[![CI](https://github.com/ai-pavel/perceptron/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-pavel/perceptron/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ai-pavel/perceptron/branch/main/graph/badge.svg)](https://codecov.io/gh/ai-pavel/perceptron)

A neural network framework built from scratch using only NumPy.

## Features

- **Layers**: Dense (fully connected), ReLU, Sigmoid, Softmax, Dropout
- **Losses**: MSE, CrossEntropy
- **Optimizers**: SGD (with optional momentum), Adam
- **Model**: Sequential with `fit()`, `predict()`, `save()`, `load()`

## Installation

```bash
pip install -e ".[dev,demo]"
```

## Quick Start

```python
from nn import Dense, ReLU, Softmax, Sequential, CrossEntropy, Adam

model = Sequential([
    Dense(784, 256),
    ReLU(),
    Dense(256, 10),
    Softmax(),
])

loss_fn = CrossEntropy()
optimizer = Adam(lr=0.001)

model.fit(x_train, y_train, loss_fn, optimizer, epochs=10, batch_size=128)
predictions = model.predict(x_test)
```

## MNIST Demo

```bash
python demo_mnist.py
```

Trains a 784-256-128-10 network with dropout and achieves >95% test accuracy.

## Tests

```bash
python -m pytest
```

## Project Structure

```
nn/
  __init__.py
  layers.py       # Layer, Dense, ReLU, Sigmoid, Softmax, Dropout
  losses.py       # MSE, CrossEntropy
  optimizers.py   # SGD, Adam
  model.py        # Sequential
tests/
demo_mnist.py
pyproject.toml
```
