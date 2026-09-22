import keras
from keras import ops
import numpy as np
import pytest

from keras_sciml.models import PINN


def test_pinn_smoke():
    model = PINN(output_dim=2, hidden_layers=3, hidden_units=32, activation="tanh")
    # Spatial coordinates: (batch=10, 2)
    coords = ops.ones((10, 2))
    u = model(coords)
    assert u.shape == (10, 2)


def test_pinn_fourier_features_smoke():
    model = PINN(
        output_dim=1,
        hidden_layers=2,
        hidden_units=32,
        use_fourier_features=True,
        fourier_dim=16,
    )
    coords = ops.ones((8, 1))
    u = model(coords)
    assert u.shape == (8, 1)


def test_pinn_residual_attachment_smoke():
    def dummy_residual(u, coords):
        return u - 1.0

    model = PINN(output_dim=1, hidden_layers=2, hidden_units=16, residual_fn=dummy_residual, pde_weight=0.5)
    coords = ops.ones((4, 2))
    # In training=True, adds PDE residual loss
    u = model(coords, training=True)
    assert len(model.losses) == 1


def test_pinn_serialization():
    model = PINN(output_dim=1, hidden_layers=2, hidden_units=16, use_fourier_features=True, fourier_dim=8)
    coords = ops.ones((2, 1))
    out1 = model(coords)

    config = keras.saving.serialize_keras_object(model)
    recovered = keras.saving.deserialize_keras_object(config)
    recovered(coords)
    recovered.set_weights(model.get_weights())
    out2 = recovered(coords)

    np.testing.assert_allclose(ops.convert_to_numpy(out1), ops.convert_to_numpy(out2), rtol=1e-5)

