import keras
from keras import ops
import numpy as np
import pytest

from keras_sciml.models import PINO, FNO1D


def test_pino_smoke():
    # Auto-instantiated FNO backbone
    model = PINO(modes=6, width=16, depth=2, out_channels=1)
    x = ops.ones((2, 16, 1))
    u = model(x)
    assert u.shape == (2, 16, 1)


def test_pino_custom_backbone_and_residual_smoke():
    fno_backbone = FNO1D(modes=4, width=12, depth=2, out_channels=1)

    def dummy_pde(u, inputs):
        return u - inputs

    model = PINO(operator_model=fno_backbone, residual_fn=dummy_pde, pde_weight=0.5, return_residual=True)
    x = ops.ones((2, 16, 1))
    u, res = model(x, training=True)
    assert u.shape == (2, 16, 1)
    assert res.shape == (2, 16, 1)
    assert len(model.losses) == 1


def test_pino_serialization():
    model = PINO(modes=4, width=8, depth=2, out_channels=1)
    x = ops.ones((1, 12, 1))
    out1 = model(x)

    config = keras.saving.serialize_keras_object(model)
    recovered = keras.saving.deserialize_keras_object(config)
    recovered(x)
    recovered.set_weights(model.get_weights())
    out2 = recovered(x)

    np.testing.assert_allclose(ops.convert_to_numpy(out1), ops.convert_to_numpy(out2), rtol=1e-5)

