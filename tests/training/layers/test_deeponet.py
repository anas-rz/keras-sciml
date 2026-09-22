import numpy as np
import pytest
import keras
from keras import layers, ops

from keras_sciml.layers.operators import (
    DeepONet,
    DeepONetCartesianProduct,
    DeepONetPointwise,
)


def test_deeponet_cartesian_learns_antiderivative():
    keras.utils.set_random_seed(42)
    np.random.seed(42)

    # Problem: s(y) = \int_0^y u(x) dx
    # u(x) = sin(omega * pi * x) -> s(y) = (1 - cos(omega * pi * y)) / (omega * pi)
    num_samples = 64
    m_sensors = 20
    n_points = 25

    x_sensors = np.linspace(0.0, 1.0, m_sensors, dtype="float32")
    y_coords = np.linspace(0.0, 1.0, n_points, dtype="float32")[:, None]

    u_list = []
    s_list = []
    for _ in range(num_samples):
        omega = np.random.uniform(1.0, 3.0)
        u_val = np.sin(omega * np.pi * x_sensors)
        s_val = (1.0 - np.cos(omega * np.pi * y_coords[:, 0])) / (omega * np.pi)
        u_list.append(u_val)
        s_list.append(s_val)

    u_train = np.array(u_list, dtype="float32")  # (num_samples, m_sensors)
    s_train = np.array(s_list, dtype="float32")  # (num_samples, n_points)

    # Tile coordinates across batch: (num_samples, n_points, 1)
    y_train = np.tile(y_coords[None, :, :], (num_samples, 1, 1))

    # Architecture
    u_in = layers.Input(shape=(m_sensors,), name="sensors")
    y_in = layers.Input(shape=(n_points, 1), name="coordinates")

    b = layers.Dense(32, activation="tanh")(u_in)
    b = layers.Dense(16)(b)

    t = layers.Dense(32, activation="tanh")(y_in)
    t = layers.Dense(16)(t)

    merger = DeepONetCartesianProduct(use_bias=True)
    out = merger([b, t])

    model = keras.Model(inputs=[u_in, y_in], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    initial_bias = np.copy(ops.convert_to_numpy(merger.bias))
    initial_loss = float(model.evaluate([u_train, y_train], s_train, verbose=0))

    history = model.fit([u_train, y_train], s_train, epochs=35, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]
    final_bias = ops.convert_to_numpy(merger.bias)

    assert final_loss < initial_loss * 0.25
    assert final_loss < 0.05
    assert np.abs(final_bias - initial_bias) > 1e-4


def test_deeponet_pointwise_learns():
    keras.utils.set_random_seed(123)
    np.random.seed(123)

    num_samples = 64
    m_features = 8

    # Aligned point-wise problem
    u_train = np.random.uniform(-1.0, 1.0, size=(num_samples, m_features)).astype("float32")
    y_train = np.random.uniform(-1.0, 1.0, size=(num_samples, 2)).astype("float32")
    # Target is simple nonlinear combination
    target = (np.sum(u_train[:, :2] * y_train, axis=-1, keepdims=True)).astype("float32")

    u_in = layers.Input(shape=(m_features,))
    y_in = layers.Input(shape=(2,))

    branch = layers.Dense(8, activation="relu")(u_in)
    trunk = layers.Dense(8, activation="relu")(y_in)

    merger = DeepONetPointwise(use_bias=True)
    out = merger([branch, trunk])

    model = keras.Model(inputs=[u_in, y_in], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.03), loss="mse")

    initial_loss = float(model.evaluate([u_train, y_train], target, verbose=0))
    history = model.fit([u_train, y_train], target, epochs=30, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]

    assert final_loss < initial_loss * 0.3

