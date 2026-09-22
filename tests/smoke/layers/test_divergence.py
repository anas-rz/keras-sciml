import tempfile
import pytest
import keras
from keras import layers, ops
import numpy as np

from keras_sciml.layers.differential import Divergence


def test_instantiation():
    layer = Divergence(dx=0.1, boundary_condition="central", keepdims=True)
    assert layer.boundary_condition == "central"
    assert layer.keepdims is True

    with pytest.raises(ValueError, match="must be 'central' or 'periodic'"):
        Divergence(boundary_condition="invalid")


def test_shapes_1d():
    # 1D: (batch, length, 1) -> (batch, length, 1)
    layer = Divergence(dx=0.05, keepdims=True)
    v = ops.ones((2, 30, 1))
    out = layer(v)
    assert out.shape == (2, 30, 1)
    assert layer.compute_output_shape((2, 30, 1)) == (2, 30, 1)

    layer_no_keep = Divergence(dx=0.05, keepdims=False)
    out_no_keep = layer_no_keep(v)
    assert out_no_keep.shape == (2, 30)
    assert layer_no_keep.compute_output_shape((2, 30, 1)) == (2, 30)


def test_shapes_2d():
    # 2D: (batch, h, w, 2) -> (batch, h, w, 1)
    layer = Divergence(dx=[0.1, 0.2], keepdims=True)
    v = ops.ones((2, 16, 16, 2))
    out = layer(v)
    assert out.shape == (2, 16, 16, 1)
    assert layer.compute_output_shape((2, 16, 16, 2)) == (2, 16, 16, 1)


def test_shapes_3d():
    # 3D: (batch, d, h, w, 3) -> (batch, d, h, w, 1)
    layer = Divergence(dx=0.1, keepdims=True)
    v = ops.ones((1, 8, 8, 8, 3))
    out = layer(v)
    assert out.shape == (1, 8, 8, 8, 1)
    assert layer.compute_output_shape((1, 8, 8, 8, 3)) == (1, 8, 8, 8, 1)


def test_multi_field_divergence():
    # 2D with 2 vector fields -> 4 channels (u1, v1, u2, v2) -> output 2 channels
    layer = Divergence(dx=[0.1, 0.1], keepdims=True)
    v = ops.ones((2, 12, 12, 4))
    out = layer(v)
    assert out.shape == (2, 12, 12, 2)
    assert layer.compute_output_shape((2, 12, 12, 4)) == (2, 12, 12, 2)


def test_incompatible_channels_error():
    # 2D requires channels to be divisible by 2
    layer = Divergence(dx=0.1)
    v = ops.ones((2, 12, 12, 3))  # 3 is not divisible by 2
    with pytest.raises(ValueError, match="must be divisible"):
        layer(v)


def test_backprop_through_divergence():
    # Incompressibility loss: ||div(v)||^2
    inputs = layers.Input(shape=(16, 16, 2))
    div = Divergence(dx=0.1, boundary_condition="periodic")(inputs)
    model = keras.Model(inputs=inputs, outputs=div)

    v_test = ops.ones((2, 16, 16, 2))
    out = model(v_test)
    assert out.shape == (2, 16, 16, 1)


def test_serialization_config():
    layer = Divergence(dx=[0.1, 0.2], boundary_condition="periodic", keepdims=False)
    cfg = layer.get_config()
    rec = Divergence.from_config(cfg)

    assert rec.dx == [0.1, 0.2]
    assert rec.boundary_condition == "periodic"
    assert rec.keepdims is False


def test_model_save_and_load():
    inputs = layers.Input(shape=(12, 12, 2))
    out = Divergence(dx=0.1, boundary_condition="central")(inputs)
    model = keras.Model(inputs=inputs, outputs=out)

    test_input = np.random.uniform(size=(2, 12, 12, 2)).astype("float32")
    orig_pred = model(test_input)

    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        model.save(tmp.name)
        loaded = keras.models.load_model(tmp.name)
        loaded_pred = loaded(test_input)

    np.testing.assert_allclose(
        ops.convert_to_numpy(orig_pred),
        ops.convert_to_numpy(loaded_pred),
        atol=1e-6,
    )

