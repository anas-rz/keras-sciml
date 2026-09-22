import numpy as np
import pytest
import keras
from keras import layers, ops

from keras_sciml.layers.operators import (
    SpectralConv1D,
    SpectralConv2D,
)


def test_spectral_conv1d_learns_operator():
    keras.utils.set_random_seed(42)
    np.random.seed(42)

    num_samples = 64
    resolution = 32
    grid = np.linspace(0.0, 1.0, resolution, dtype="float32")

    # Generate synthetic 1D input functions and transformed outputs
    x_data = []
    y_data = []
    for _ in range(num_samples):
        freq = np.random.choice([1, 2, 3])
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        u = np.sin(2.0 * np.pi * freq * grid + phase)
        # Operator mapping: low-pass / scaled frequency response
        v = (1.0 / (1.0 + freq)) * np.cos(2.0 * np.pi * freq * grid + phase)
        x_data.append(u[:, None])
        y_data.append(v[:, None])

    x_train = np.array(x_data, dtype="float32")
    y_train = np.array(y_data, dtype="float32")

    # Small FNO block
    inputs = layers.Input(shape=(resolution, 1))
    x = SpectralConv1D(filters=8, modes=6, use_bias=True)(inputs)
    x = layers.Activation("gelu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    initial_loss = float(model.evaluate(x_train, y_train, verbose=0))
    history = model.fit(x_train, y_train, epochs=30, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]

    assert final_loss < initial_loss * 0.3
    assert final_loss < 0.1


def test_spectral_conv2d_learns_operator():
    keras.utils.set_random_seed(123)
    np.random.seed(123)

    num_samples = 32
    h, w = 12, 12
    x_grid = np.linspace(0.0, 1.0, h, dtype="float32")
    y_grid = np.linspace(0.0, 1.0, w, dtype="float32")
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")

    x_data = []
    y_data = []
    for _ in range(num_samples):
        freq = np.random.choice([1, 2])
        u = np.sin(2.0 * np.pi * freq * xx) * np.cos(2.0 * np.pi * freq * yy)
        v = (0.5 / freq) * np.sin(2.0 * np.pi * freq * xx) * np.cos(2.0 * np.pi * freq * yy)
        x_data.append(u[:, :, None])
        y_data.append(v[:, :, None])

    x_train = np.array(x_data, dtype="float32")
    y_train = np.array(y_data, dtype="float32")

    inputs = layers.Input(shape=(h, w, 1))
    layer = SpectralConv2D(filters=4, modes=(3, 3), use_bias=True)
    x = layer(inputs)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.03), loss="mse")

    initial_w1_r = np.copy(ops.convert_to_numpy(layer.kernel1_real))
    initial_loss = float(model.evaluate(x_train, y_train, verbose=0))

    history = model.fit(x_train, y_train, epochs=25, batch_size=8, verbose=0)
    final_loss = history.history["loss"][-1]
    final_w1_r = ops.convert_to_numpy(layer.kernel1_real)

    assert final_loss < initial_loss * 0.3
    assert np.max(np.abs(final_w1_r - initial_w1_r)) > 1e-4

