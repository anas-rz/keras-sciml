import tempfile
import pytest
import keras
from keras import ops
import numpy as np

from keras_sciml.layers.operators import (
    SpectralConv,
    SpectralConv1D,
    SpectralConv2D,
)


def test_instantiation_1d():
    layer = SpectralConv1D(filters=8, modes=4, use_bias=True)
    assert layer.filters == 8
    assert layer.modes == 4
    assert layer.use_bias is True

    with pytest.raises(ValueError, match="positive integer"):
        SpectralConv1D(filters=0, modes=4)

    with pytest.raises(ValueError, match="positive integer"):
        SpectralConv1D(filters=8, modes=-1)


def test_instantiation_2d():
    layer = SpectralConv2D(filters=16, modes=(4, 6), use_bias=False)
    assert layer.filters == 16
    assert layer.modes == (4, 6)
    assert layer.use_bias is False

    layer_int = SpectralConv2D(filters=8, modes=5)
    assert layer_int.modes == (5, 5)

    with pytest.raises(ValueError, match="tuple/list of two integers"):
        SpectralConv2D(filters=8, modes=(4, 5, 6))


def test_shapes_1d():
    batch, length, in_channels, out_channels = 2, 32, 4, 8
    layer = SpectralConv1D(filters=out_channels, modes=8, use_bias=True)
    x = ops.ones((batch, length, in_channels))
    y = layer(x)
    assert y.shape == (batch, length, out_channels)
    assert layer.compute_output_shape((batch, length, in_channels)) == (batch, length, out_channels)
    assert len(layer.trainable_weights) == 3  # kernel_real, kernel_imag, bias


def test_shapes_2d():
    batch, h, w, in_channels, out_channels = 2, 16, 16, 3, 6
    layer = SpectralConv2D(filters=out_channels, modes=(4, 4), use_bias=True)
    x = ops.ones((batch, h, w, in_channels))
    y = layer(x)
    assert y.shape == (batch, h, w, out_channels)
    assert layer.compute_output_shape((batch, h, w, in_channels)) == (batch, h, w, out_channels)
    assert len(layer.trainable_weights) == 5  # kernel1_r, kernel1_i, kernel2_r, kernel2_i, bias


def test_unified_spectral_conv_dispatch():
    # 1D auto-dispatch
    conv1d = SpectralConv(filters=4, modes=4)
    x1d = ops.ones((2, 16, 3))
    y1d = conv1d(x1d)
    assert y1d.shape == (2, 16, 4)
    assert isinstance(conv1d.sublayer, SpectralConv1D)

    # 2D auto-dispatch
    conv2d = SpectralConv(filters=4, modes=(4, 4))
    x2d = ops.ones((2, 16, 16, 3))
    y2d = conv2d(x2d)
    assert y2d.shape == (2, 16, 16, 4)
    assert isinstance(conv2d.sublayer, SpectralConv2D)


def test_serialization_1d_and_2d():
    # 1D
    layer1d = SpectralConv1D(filters=8, modes=4, use_bias=True)
    cfg1d = layer1d.get_config()
    rec1d = SpectralConv1D.from_config(cfg1d)
    assert rec1d.filters == 8
    assert rec1d.modes == 4
    assert rec1d.use_bias is True

    # 2D
    layer2d = SpectralConv2D(filters=16, modes=(4, 8), use_bias=False)
    cfg2d = layer2d.get_config()
    rec2d = SpectralConv2D.from_config(cfg2d)
    assert rec2d.filters == 16
    assert rec2d.modes == (4, 8)
    assert rec2d.use_bias is False


def test_model_save_and_load_roundtrip_1d():
    inputs = keras.layers.Input(shape=(16, 2))
    x = SpectralConv1D(filters=4, modes=4, use_bias=True)(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    test_input = np.random.uniform(size=(2, 16, 2)).astype("float32")
    original_pred = model(test_input)

    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        model.save(tmp.name)
        loaded_model = keras.models.load_model(tmp.name)
        loaded_pred = loaded_model(test_input)

    np.testing.assert_allclose(
        ops.convert_to_numpy(original_pred),
        ops.convert_to_numpy(loaded_pred),
        atol=1e-5,
    )


def test_model_save_and_load_roundtrip_2d():
    inputs = keras.layers.Input(shape=(12, 12, 2))
    x = SpectralConv2D(filters=4, modes=(3, 3), use_bias=True)(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    test_input = np.random.uniform(size=(2, 12, 12, 2)).astype("float32")
    original_pred = model(test_input)

    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        model.save(tmp.name)
        loaded_model = keras.models.load_model(tmp.name)
        loaded_pred = loaded_model(test_input)

    np.testing.assert_allclose(
        ops.convert_to_numpy(original_pred),
        ops.convert_to_numpy(loaded_pred),
        atol=1e-5,
    )

