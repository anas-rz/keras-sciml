import numpy as np
import pytest
import keras
from keras import layers, ops

from keras_sciml.layers.physics import ConservationConstraint, PositivityConstraint


def test_positivity_constraint_learns_soft():
    # Target function has negative values: y = sin(2*pi*x) on [0, 1]
    # Positivity constraint penalizes negative outputs
    keras.utils.set_random_seed(42)
    np.random.seed(42)

    n_points = 20
    x_coords = np.linspace(0.0, 1.0, n_points, dtype="float32").reshape(1, n_points, 1)
    # Target has negative values
    y_target = np.sin(2.0 * np.pi * x_coords).astype("float32")

    inputs = layers.Input(shape=(n_points, 1))
    h = layers.Flatten()(inputs)
    h = layers.Dense(32, activation="relu")(h)
    u_pred = layers.Reshape((n_points, 1))(layers.Dense(n_points)(h))
    outputs = PositivityConstraint(min_value=0.0, weight=30.0, add_loss=True)(u_pred)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    # Train for 50 epochs
    _ = model.fit(x_coords, y_target, epochs=50, verbose=0)

    pred = ops.convert_to_numpy(model(x_coords))
    # Target goes down to -1.0; penalty strongly suppresses negatives
    assert np.min(pred) >= -0.15
    assert np.min(pred) > np.min(y_target) + 0.8



def test_positivity_constraint_hard():
    keras.utils.set_random_seed(42)
    n_points = 15
    x_coords = np.linspace(0.0, 1.0, n_points, dtype="float32").reshape(1, n_points, 1)
    y_target = -np.ones_like(x_coords)  # all negative target

    inputs = layers.Input(shape=(n_points, 1))
    h = layers.Flatten()(inputs)
    h = layers.Dense(16, activation="relu")(h)
    u_pred = layers.Reshape((n_points, 1))(layers.Dense(n_points)(h))
    outputs = PositivityConstraint(min_value=0.0, enforcement="hard")(u_pred)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

    _ = model.fit(x_coords, y_target, epochs=10, verbose=0)

    pred = ops.convert_to_numpy(model(x_coords))
    # Hard enforcement guarantees all values are >= 0.0 strictly
    assert np.all(pred >= 0.0)


def test_conservation_constraint_hard():
    # Conserve total mass integral = 2.0
    keras.utils.set_random_seed(42)
    n_points = 20
    dx = 1.0 / n_points
    x_coords = np.linspace(0.0, 1.0, n_points, dtype="float32").reshape(1, n_points, 1)
    y_target = np.ones_like(x_coords) * 2.0

    inputs = layers.Input(shape=(n_points, 1))
    h = layers.Flatten()(inputs)
    h = layers.Dense(16, activation="relu")(h)
    u_pred = layers.Reshape((n_points, 1))(layers.Dense(n_points)(h))
    outputs = ConservationConstraint(target=2.0, reduction="integral", dx=dx, enforcement="hard")(u_pred)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

    # Before and after training, integral must strictly be 2.0
    pred_init = ops.convert_to_numpy(model(x_coords))
    integral_init = np.sum(pred_init) * dx
    np.testing.assert_allclose(integral_init, 2.0, atol=1e-5)

    _ = model.fit(x_coords, y_target, epochs=15, verbose=0)

    pred_final = ops.convert_to_numpy(model(x_coords))
    integral_final = np.sum(pred_final) * dx
    np.testing.assert_allclose(integral_final, 2.0, atol=1e-5)
