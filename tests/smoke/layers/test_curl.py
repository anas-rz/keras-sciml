import tempfile
import pytest
import keras
from keras import layers, ops
import numpy as np

from keras_sciml.layers.differential import Curl


def test_instantiation():
    layer = Curl(dx=0.1, boundary_condition="central", keepdims=True)
    assert layer.boundary_condition == "central"
    assert layer.keepdims is True

    with pytest.raises(ValueError, match="must be 'central' or 'periodic'"):
        Curl(boundary_condition="invalid")


def test_shapes_2d():
    # 2D: (batch, h, w, 2) -> (batch, h, w, 1) or (batch, h, w)
    layer = Curl(dx=[0.1, 0.2], keepdims=True)
    v = ops.ones((2, 16, 16, 2))
    out = layer(v)
    assert out.shape == (2, 16, 16, 1)
    assert layer.compute_output_shape((2, 16, 16, 2)) == (2, 16, 16, 1)

    layer_no_keep = Curl(dx=[0.1, 0.2], keepdims=False)
    out_no_keep = layer_no_keep(v)
    assert out_no_keep.shape == (2, 16, 16)
    assert layer_no_keep.compute_output_shape((2, 16, 16, 2)) == (2, 16, 16)


def test_shapes_3d():
    # 3D: (batch, d, h, w, 3) -> (batch, d, h, w, 3)
    layer = Curl(dx=[0.1, 0.1, 0.2])
    v = ops.ones((1, 8, 8, 8, 3))
    out = layer(v)
    assert out.shape == (1, 8, 8, 8, 3)
    assert layer.compute_output_shape((1, 8, 8, 8, 3)) == (1, 8, 8, 8, 3)


def test_multi_field_curl_2d():
    # 2D with 2 vector fields -> 4 channels (u1, v1, u2, v2) -> output 2 channels
    layer = Curl(dx=[0.1, 0.1], keepdims=True)
    v = ops.ones((2, 12, 12, 4))
    out = layer(v)
    assert out.shape == (2, 12, 12, 2)
    assert layer.compute_output_shape((2, 12, 12, 4)) == (2, 12, 12, 2)


def test_multi_field_curl_3d():
    # 3D with 2 vector fields -> 6 channels -> output 6 channels
    layer = Curl(dx=0.1)
    v = ops.ones((2, 8, 8, 8, 6))
    out = layer(v)
    assert out.shape == (2, 8, 8, 8, 6)
    assert layer.compute_output_shape((2, 8, 8, 8, 6)) == (2, 8, 8, 8, 6)


def test_invalid_spatial_dimensions():
    layer = Curl()
    v_1d = ops.ones((2, 10, 1))
    with pytest.raises(ValueError, match="only defined for 2D and 3D"):
        layer(v_1d)

    v_4d = ops.ones((2, 4, 4, 4, 4, 4))
    with pytest.raises(ValueError, match="only defined for 2D and 3D"):
        layer(v_4d)


def test_incompatible_channels_error():
    # 2D requires channels to be divisible by 2
    layer_2d = Curl(dx=0.1)
    v_2d = ops.ones((2, 12, 12, 3))
    with pytest.raises(ValueError, match="must be divisible"):
        layer_2d(v_2d)

    # 3D requires channels to be divisible by 3
    layer_3d = Curl(dx=0.1)
    v_3d = ops.ones((2, 8, 8, 8, 4))
    with pytest.raises(ValueError, match="must be divisible"):
        layer_3d(v_3d)


def test_backprop_through_curl():
    # Vorticity loss: ||curl(v)||^2
    inputs = layers.Input(shape=(16, 16, 2))
    curl = Curl(dx=0.1, boundary_condition="periodic")(inputs)
    model = keras.Model(inputs=inputs, outputs=curl)

    v_test = ops.ones((2, 16, 16, 2))
    out = model(v_test)
    assert out.shape == (2, 16, 16, 1)


def test_serialization_config():
    layer = Curl(dx=[0.1, 0.2], boundary_condition="periodic", keepdims=False)
    cfg = layer.get_config()
    rec = Curl.from_config(cfg)

    assert rec.dx == [0.1, 0.2]
    assert rec.boundary_condition == "periodic"
    assert rec.keepdims is False


def test_model_save_and_load():
    inputs = layers.Input(shape=(12, 12, 2))
    out = Curl(dx=0.1, boundary_condition="central")(inputs)
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

