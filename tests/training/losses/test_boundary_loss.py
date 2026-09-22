import keras
from keras import layers, ops
import numpy as np
import pytest

from keras_sciml.losses import BoundaryLoss


def test_boundary_loss_training():
    keras.utils.set_random_seed(42)

    # Train a network so that boundary values on a 1D grid equal 0.0 (homogeneous Dirichlet)
    # and interior matches target or minimizes boundary loss
    nx = 16
    batch_size = 8
    x_in = np.ones((batch_size, 1), dtype=np.float32)

    # Network initialized with non-zero output
    model = keras.Sequential([
        layers.Dense(32, activation="relu"),
        layers.Dense(nx),
        layers.Reshape((nx, 1)),
    ])

    bl = BoundaryLoss(bc_type="dirichlet", value=0.0)
    # Target values can be dummy zeros since BoundaryLoss with value=0.0 penalizes predicted boundaries
    y_dummy = np.zeros((batch_size, nx, 1), dtype=np.float32)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.03), loss=bl)

    initial_loss = float(model.evaluate(x_in, y_dummy, verbose=0))
    history = model.fit(x_in, y_dummy, epochs=50, batch_size=batch_size, verbose=0)
    final_loss = history.history["loss"][-1]

    assert final_loss < initial_loss * 0.1
    # Check boundaries on predictions
    preds = model.predict(x_in, verbose=0)
    np.testing.assert_allclose(preds[:, 0, 0], np.zeros(batch_size), atol=0.05)
    np.testing.assert_allclose(preds[:, -1, 0], np.zeros(batch_size), atol=0.05)

