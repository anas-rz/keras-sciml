import keras
from keras import layers, ops
import numpy as np
import pytest

from keras_sciml.losses import H1Loss, RelativeL2Loss


def test_relative_l2_training():
    keras.utils.set_random_seed(42)

    # Operator learning: map input parameters to 1D continuous field
    n_samples = 64
    nx = 32
    x = np.linspace(0, 1, nx, dtype=np.float32)

    # Random parameters (amplitude, frequency)
    rng = np.random.default_rng(42)
    params = rng.uniform(0.8, 1.5, (n_samples, 2)).astype(np.float32)

    y_train = np.stack([
        params[i, 0] * np.sin(np.pi * params[i, 1] * x)
        for i in range(n_samples)
    ])[:, :, None]

    model = keras.Sequential([
        layers.Dense(64, activation="gelu"),
        layers.Dense(64, activation="gelu"),
        layers.Dense(nx),
        layers.Reshape((nx, 1)),
    ])

    loss_fn = RelativeL2Loss()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)

    initial_loss = float(model.evaluate(params, y_train, verbose=0))
    history = model.fit(params, y_train, epochs=80, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]

    # Verify significant learning
    assert final_loss < initial_loss * 0.3
    assert final_loss < 0.25


def test_h1_loss_training():
    keras.utils.set_random_seed(42)

    nx = 20
    batch_size = 8
    dx = 1.0 / (nx - 1)
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y_train = np.stack([np.sin(np.pi * x) for _ in range(batch_size)])[:, :, None]
    x_in = np.ones((batch_size, 1), dtype=np.float32)

    model = keras.Sequential([
        layers.Dense(32, activation="tanh"),
        layers.Dense(nx),
        layers.Reshape((nx, 1)),
    ])

    loss_fn = H1Loss(alpha=0.1, dx=dx)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss=loss_fn)

    initial_loss = float(model.evaluate(x_in, y_train, verbose=0))
    history = model.fit(x_in, y_train, epochs=60, batch_size=batch_size, verbose=0)
    final_loss = history.history["loss"][-1]

    assert final_loss < initial_loss * 0.2

