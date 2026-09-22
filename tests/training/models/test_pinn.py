import keras
from keras import ops
import numpy as np
import pytest

from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import PINN


def test_pinn_learns():
    keras.utils.set_random_seed(42)

    # Learn 1D function u(x) = sin(2*pi*x) with PINN backbone
    n_points = 64
    x = np.linspace(0, 1, n_points, dtype=np.float32)[:, None]
    u_true = np.sin(2.0 * np.pi * x)

    model = PINN(output_dim=1, hidden_layers=3, hidden_units=32, activation="tanh")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    initial_loss = float(model.evaluate(x, u_true, verbose=0))
    history = model.fit(x, u_true, epochs=60, batch_size=32, verbose=0)
    final_loss = history.history["loss"][-1]

    # Verify significant learning
    assert final_loss < initial_loss * 0.2
    assert final_loss < 0.05
