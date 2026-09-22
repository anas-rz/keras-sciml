import numpy as np
import pytest
import keras
from keras import layers, ops

from keras_sciml.layers.physics import DirichletBC, PoissonResidual


def test_poisson_residual_learns():
    # Solve 1D Poisson equation: -u''(x) = pi^2 * sin(pi*x) with u(0)=0, u(1)=0
    # Exact solution: u(x) = sin(pi*x)
    keras.utils.set_random_seed(42)
    np.random.seed(42)

    n_points = 21
    dx = 1.0 / (n_points - 1)
    x_coords = np.linspace(0.0, 1.0, n_points, dtype="float32").reshape(1, n_points, 1)
    f_forcing = (np.pi**2 * np.sin(np.pi * x_coords)).astype("float32")
    exact_sol = np.sin(np.pi * x_coords).astype("float32")

    inputs = layers.Input(shape=(n_points, 1))
    h = layers.Flatten()(inputs)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dense(64, activation="relu")(h)
    u_pred = layers.Reshape((n_points, 1))(layers.Dense(n_points)(h))

    # Apply exact hard Dirichlet BC: u(0) = 0, u(1) = 0
    u_bc = DirichletBC(value=0.0, enforcement="hard")(u_pred)

    # Attach Poisson PDE residual loss
    u_out = PoissonResidual(dx=dx, forcing=f_forcing, weight=1e-4, add_loss=True)(u_bc)

    model = keras.Model(inputs=inputs, outputs=u_out)
    # We fit exact target as supervised loss, regularized by the PDE residual
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

    initial_loss = float(model.evaluate(x_coords, exact_sol, verbose=0))

    # Train for 50 epochs
    history = model.fit(x_coords, exact_sol, epochs=50, verbose=0)
    final_loss = float(history.history["loss"][-1])

    # Assert loss decreased significantly
    assert final_loss < initial_loss * 0.1

    # Check predicted solution vs exact
    pred = ops.convert_to_numpy(model(x_coords))
    l2_error = np.mean(np.square(pred - exact_sol))
    assert l2_error < 0.35
