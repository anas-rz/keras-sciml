import numpy as np
import pytest
import keras
from keras import layers, ops

from keras_sciml.layers.physics import DirichletBC


def test_boundary_condition_learns_soft():
    # Fit target function u(x) = sin(pi * x) on [0, 1] with DirichletBC u(0)=0, u(1)=0
    keras.utils.set_random_seed(42)
    np.random.seed(42)

    n_points = 21
    x_coords = np.linspace(0.0, 1.0, n_points, dtype="float32").reshape(1, n_points, 1)
    y_target = np.sin(np.pi * x_coords).astype("float32")

    # Build model: Dense layers over grid + DirichletBC
    inputs = layers.Input(shape=(n_points, 1))
    h = layers.Flatten()(inputs)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dense(64, activation="relu")(h)
    u_pred = layers.Reshape((n_points, 1))(layers.Dense(n_points)(h))
    outputs = DirichletBC(value=0.0, weight=5.0, add_loss=True)(u_pred)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

    # Evaluate initial loss
    initial_loss = float(model.evaluate(x_coords, y_target, verbose=0))

    # Train for 50 epochs
    history = model.fit(x_coords, y_target, epochs=50, verbose=0)
    final_loss = float(history.history["loss"][-1])

    # Assert loss decreased significantly
    assert final_loss < initial_loss * 0.1

    # Check that boundaries are well satisfied
    pred = ops.convert_to_numpy(model(x_coords))
    boundary_error = np.abs(pred[0, 0, 0]) + np.abs(pred[0, -1, 0])
    assert boundary_error < 0.05


def test_boundary_condition_hard_enforcement():
    # Hard enforcement guarantees boundary conditions are strictly 0.0 at all times
    keras.utils.set_random_seed(42)
    n_points = 15
    x_coords = np.linspace(0.0, 1.0, n_points, dtype="float32").reshape(1, n_points, 1)
    y_target = np.sin(np.pi * x_coords).astype("float32")

    inputs = layers.Input(shape=(n_points, 1))
    h = layers.Flatten()(inputs)
    h = layers.Dense(32, activation="relu")(h)
    u_pred = layers.Reshape((n_points, 1))(layers.Dense(n_points)(h))
    outputs = DirichletBC(value=0.0, enforcement="hard")(u_pred)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.03), loss="mse")

    # Check boundaries before training
    pred_init = ops.convert_to_numpy(model(x_coords))
    np.testing.assert_allclose(pred_init[0, 0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(pred_init[0, -1, 0], 0.0, atol=1e-6)

    # Train for 20 epochs
    history = model.fit(x_coords, y_target, epochs=20, verbose=0)
    final_loss = float(history.history["loss"][-1])

    # Check boundaries after training
    pred_final = ops.convert_to_numpy(model(x_coords))
    np.testing.assert_allclose(pred_final[0, 0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(pred_final[0, -1, 0], 0.0, atol=1e-6)
