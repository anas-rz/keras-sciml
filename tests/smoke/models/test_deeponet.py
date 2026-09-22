import keras
from keras import ops
import numpy as np
import pytest

from keras_sciml.models import DeepONet


def test_deeponet_cartesian_smoke():
    model = DeepONet(
        latent_dim=16,
        hidden_layers=2,
        hidden_units=32,
        mode="cartesian",
    )
    # Branch inputs: (batch=4, sensors=20)
    # Trunk inputs: (coords=50, dim=2)
    u = ops.ones((4, 20))
    y = ops.ones((50, 2))
    out = model([u, y])
    assert out.shape == (4, 50)


def test_deeponet_pointwise_smoke():
    model = DeepONet(
        latent_dim=16,
        hidden_layers=2,
        hidden_units=32,
        mode="pointwise",
    )
    # Aligned pairs: (batch=5, sensors=10), (batch=5, dim=1)
    u = ops.ones((5, 10))
    y = ops.ones((5, 1))
    out = model([u, y])
    assert out.shape == (5, 1)


def test_deeponet_serialization():
    model = DeepONet(latent_dim=8, hidden_layers=2, hidden_units=16)
    u = ops.ones((2, 10))
    y = ops.ones((15, 1))
    out1 = model([u, y])

    config = keras.saving.serialize_keras_object(model)
    recovered = keras.saving.deserialize_keras_object(config)
    recovered([u, y])
    recovered.set_weights(model.get_weights())
    out2 = recovered([u, y])

    np.testing.assert_allclose(ops.convert_to_numpy(out1), ops.convert_to_numpy(out2), rtol=1e-5)

