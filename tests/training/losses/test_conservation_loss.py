import keras
from keras import layers, ops
import numpy as np
import pytest

from keras_sciml.losses import ConservationLoss


def test_conservation_loss_training():
    keras.utils.set_random_seed(42)

    # Train a model so that predicted field preserves a target integral
    nx = 16
    batch_size = 8
    dx = 0.1
    target_integral = 5.0

    x_in = np.ones((batch_size, 1), dtype=np.float32)
    y_dummy = np.zeros((batch_size, nx, 1), dtype=np.float32)

    model = keras.Sequential([
        layers.Dense(32, activation="relu"),
        layers.Dense(nx),
        layers.Reshape((nx, 1)),
    ])

    cl = ConservationLoss(reduction_type="integral", target=target_integral, dx=dx)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.03), loss=cl)

    initial_loss = float(model.evaluate(x_in, y_dummy, verbose=0))
    history = model.fit(x_in, y_dummy, epochs=80, batch_size=batch_size, verbose=0)
    final_loss = history.history["loss"][-1]

    assert final_loss < initial_loss * 0.1

    # Check predicted integral
    preds = model.predict(x_in, verbose=0)
    pred_integrals = np.sum(preds, axis=(1, 2)) * dx
    np.testing.assert_allclose(pred_integrals, np.full(batch_size, target_integral), atol=0.25)
