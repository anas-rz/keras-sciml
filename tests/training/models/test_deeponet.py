import keras
from keras import ops
import numpy as np
import pytest

from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import DeepONet


def test_deeponet_learns():
    keras.utils.set_random_seed(42)

    # Operator learning: antiderivative G(u)(y) = int_0^y u(s) ds
    # For u(x) = c * cos(pi * x), exact antiderivative is G(u)(y) = c / pi * sin(pi * y)
    n_samples = 32
    m_sensors = 20
    n_points = 25

    x_sensors = np.linspace(0, 1, m_sensors, dtype=np.float32)
    y_coords = np.linspace(0, 1, n_points, dtype=np.float32)[:, None]

    rng = np.random.default_rng(42)
    c = rng.uniform(0.5, 2.0, (n_samples, 1)).astype(np.float32)

    # Branch input: (batch, m_sensors)
    u_branch = c * np.cos(np.pi * x_sensors[None, :])
    # Trunk input: (batch, n_points, 1)
    y_batch = np.tile(y_coords[None, :, :], (n_samples, 1, 1))
    # Target: (batch, n_points)
    G_target = (c / np.pi) * np.sin(np.pi * y_coords.T)

    model = DeepONet(
        latent_dim=16,
        hidden_layers=2,
        hidden_units=32,
        activation="tanh",
        mode="cartesian",
    )

    loss_fn = RelativeL2Loss()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss=loss_fn)

    initial_loss = float(model.evaluate([u_branch, y_batch], G_target, verbose=0))
    history = model.fit([u_branch, y_batch], G_target, epochs=80, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]

    # Verify significant learning
    assert final_loss < initial_loss * 0.6
