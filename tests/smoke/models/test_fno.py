import keras
from keras import ops
import numpy as np
import pytest

from keras_sciml.models import FNO, FNO1D, FNO2D


def test_fno_1d_smoke():
    model = FNO1D(modes=8, width=16, depth=2, out_channels=2)
    x = ops.ones((3, 32, 1))
    y = model(x)
    assert y.shape == (3, 32, 2)


def test_fno_2d_smoke():
    model = FNO2D(modes=(6, 6), width=16, depth=2, out_channels=1)
    x = ops.ones((2, 16, 16, 3))
    y = model(x)
    assert y.shape == (2, 16, 16, 1)


def test_fno_unified_dispatch_smoke():
    # 1D input
    model_1d = FNO(modes=8, width=16, depth=2)
    x_1d = ops.ones((2, 24, 2))
    y_1d = model_1d(x_1d)
    assert y_1d.shape == (2, 24, 2)

    # 2D input matching README.md example
    model_2d = FNO(modes=8, width=32, depth=2)
    x_2d = ops.ones((2, 16, 16, 3))
    y_2d = model_2d(x_2d)
    assert y_2d.shape == (2, 16, 16, 3)


def test_fno_serialization():
    model = FNO1D(modes=4, width=8, depth=2, out_channels=1)
    x = ops.ones((1, 16, 1))
    out1 = model(x)

    config = keras.saving.serialize_keras_object(model)
    recovered = keras.saving.deserialize_keras_object(config)
    recovered(x)
    recovered.set_weights(model.get_weights())
    out2 = recovered(x)

    np.testing.assert_allclose(ops.convert_to_numpy(out1), ops.convert_to_numpy(out2), rtol=1e-5)
