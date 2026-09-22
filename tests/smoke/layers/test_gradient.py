import tempfile
import pytest
import keras
from keras import layers, ops
import numpy as np

from keras_sciml.layers.differential import Gradient


def test_instantiation():
    layer = Gradient(dx=0.1, boundary_condition="central", concat_components=True)
    assert layer.boundary_condition == "central"
    assert layer.concat_components is True

    with pytest.raises(ValueError, match="must be 'central' or 'periodic'"):
        Gradient(boundary_condition="invalid_bc")


def test_shapes_1d():
    # 1D domain: (batch, length, channels) = (2, 30, 3)
    layer = Gradient(dx=0.05, concat_components=False)
    x = ops.ones((2, 30, 3))
    out = layer(x)
    assert out.shape == (2, 30, 3, 1)
    assert layer.compute_output_shape((2, 30, 3)) == (2, 30, 3, 1)

    layer_cat = Gradient(dx=0.05, concat_components=True)
    out_cat = layer_cat(x)
    assert out_cat.shape == (2, 30, 3)
    assert layer_cat.compute_output_shape((2, 30, 3)) == (2, 30, 3)


def test_shapes_2d():
    # 2D domain: (batch, h, w, channels) = (2, 16, 16, 4)
    layer = Gradient(dx=[0.1, 0.2], concat_components=False)
    x = ops.ones((2, 16, 16, 4))
    out = layer(x)
    assert out.shape == (2, 16, 16, 4, 2)
    assert layer.compute_output_shape((2, 16, 16, 4)) == (2, 16, 16, 4, 2)

    layer_cat = Gradient(dx=[0.1, 0.2], concat_components=True)
    out_cat = layer_cat(x)
    assert out_cat.shape == (2, 16, 16, 8)
    assert layer_cat.compute_output_shape((2, 16, 16, 4)) == (2, 16, 16, 8)


def test_shapes_3d():
    # 3D domain: (batch, d, h, w, channels) = (1, 8, 8, 8, 2)
    layer = Gradient(dx=0.1, concat_components=False)
    x = ops.ones((1, 8, 8, 8, 2))
    out = layer(x)
    assert out.shape == (1, 8, 8, 8, 2, 3)
    assert layer.compute_output_shape((1, 8, 8, 8, 2)) == (1, 8, 8, 8, 2, 3)


def test_axes_selection():
    # Compute derivative only along axis 2 (width) in 2D
    layer = Gradient(dx=0.1, axes=[2], concat_components=False)
    x = ops.ones((2, 12, 14, 3))
    out = layer(x)
    assert out.shape == (2, 12, 14, 3, 1)


def test_backprop_through_gradient():
    # Test gradient of total variation loss w.r.t input
    inputs = layers.Input(shape=(16, 1))
    grad = Gradient(dx=0.1, boundary_condition="periodic", concat_components=True)(inputs)
    # Total variation loss: sum(grad^2)
    loss = ops.sum(ops.square(grad))

    model = keras.Model(inputs=inputs, outputs=grad)
    x_test = ops.convert_to_tensor(np.sin(np.linspace(0, 2 * np.pi, 16, dtype="float32")[None, :, None]))

    # Forward pass
    y = model(x_test)
    assert y.shape == (1, 16, 1)


def test_serialization_config():
    layer = Gradient(dx=[0.05, 0.1], boundary_condition="periodic", axes=[1, 2], concat_components=True)
    cfg = layer.get_config()
    rec = Gradient.from_config(cfg)

    assert rec.dx == [0.05, 0.1]
    assert rec.boundary_condition == "periodic"
    assert rec.axes == [1, 2]
    assert rec.concat_components is True


def test_model_save_and_load():
    inputs = layers.Input(shape=(12, 12, 1))
    out = Gradient(dx=0.1, boundary_condition="central", concat_components=True)(inputs)
    model = keras.Model(inputs=inputs, outputs=out)

    test_input = np.random.uniform(size=(2, 12, 12, 1)).astype("float32")
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

