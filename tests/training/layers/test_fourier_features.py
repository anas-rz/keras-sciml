import numpy as np
import pytest
import keras
from keras import layers, ops


def test_fourier_features_learns_high_frequency_1d():
    keras.utils.set_random_seed(42)
    np.random.seed(42)

    # Synthetic 1D high-frequency regression dataset
    x_train = np.linspace(-1.0, 1.0, 128, dtype="float32")[:, None]
    y_train = (np.sin(8.0 * np.pi * x_train) + 0.5 * np.cos(16.0 * np.pi * x_train)).astype("float32")

    from keras_sciml.layers.operators import FourierFeatures

    inputs = layers.Input(shape=(1,))
    x = FourierFeatures(output_dim=64, scale=10.0, trainable_frequencies=False)(inputs)
    x = layers.Dense(32, activation="gelu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.03), loss="mse")

    initial_loss = float(model.evaluate(x_train, y_train, verbose=0))
    history = model.fit(x_train, y_train, epochs=40, batch_size=32, verbose=0)
    final_loss = history.history["loss"][-1]

    # Verify model learned and loss decreased significantly
    assert final_loss < initial_loss * 0.2
    assert final_loss < 0.1


def test_fourier_features_trainable_frequencies_updates():
    keras.utils.set_random_seed(123)
    np.random.seed(123)

    x_train = np.linspace(-1.0, 1.0, 64, dtype="float32")[:, None]
    y_train = np.sin(4.0 * np.pi * x_train).astype("float32")

    from keras_sciml.layers.operators import FourierFeatures

    layer = FourierFeatures(output_dim=16, scale=1.0, trainable_frequencies=True)
    inputs = layers.Input(shape=(1,))
    x = layer(inputs)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    # Capture initial weights of frequency matrix
    initial_kernel = np.copy(ops.convert_to_numpy(layer.kernel))

    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    final_kernel = ops.convert_to_numpy(layer.kernel)

    # Verify that frequency weights actually updated during backpropagation
    weight_diff = np.max(np.abs(final_kernel - initial_kernel))
    assert weight_diff > 1e-4

