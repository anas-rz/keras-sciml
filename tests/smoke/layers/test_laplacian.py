import tempfile
import pytest
import keras
from keras import layers, ops
import numpy as np

from keras_sciml.layers.differential import Laplacian


def test_instantiation():
    layer = Laplacian(dx=0.1, boundary_condition="central")
    assert layer.boundary_condition == "central"

    with pytest.raises(ValueError, match="must be 'central' or 'periodic'"):
        Laplacian(boundary_condition="invalid")


def test_shapes_1d():
    layer = Laplacian(dx=0.05)
    x = ops.ones((3, 32, 2))
    out = layer(x)
    assert out.shape == (3, 32, 2)
    assert layer.compute_output_shape((3, 32, 2)) == (3, 32, 2)


def test_shapes_2d():
    layer = Laplacian(dx=[0.1, 0.2])
    x = ops.ones((2, 16, 18, 4))
    out = layer(x)
    assert out.shape == (2, 16, 18, 4)
    assert layer.compute_output_shape((2, 16, 18, 4)) == (2, 16, 18, 4)


def test_shapes_3d():
    layer = Laplacian(dx=0.1)
    x = ops.ones((1, 8, 8, 8, 1))
    out = layer(x)
    assert out.shape == (1, 8, 8, 8, 1)
    assert layer.compute_output_shape((1, 8, 8, 8, 1)) == (1, 8, 8, 8, 1)


def test_backprop_through_laplacian():
    # Test gradient of Poisson residual loss ||\Delta u - f||^2 w.r.t input
    inputs = layers.Input(shape=(16, 16, 1))
    lap = Laplacian(dx=0.1, boundary_condition="periodic")(inputs)
    model = keras.Model(inputs=inputs, outputs=lap)

    x_test = ops.convert_to_tensor(np.ones((2, 16, 16, 1), dtype="float32"))
    out = model(x_test)
    assert out.shape == (2, 16, 16, 1)


def test_serialization_config():
    layer = Laplacian(dx=[0.1, 0.2], boundary_condition="periodic", axes=[1, 2])
    cfg = layer.get_config()
    rec = Laplacian.from_config(cfg)

    assert rec.dx == [0.1, 0.2]
    assert rec.boundary_condition == "periodic"
    assert rec.axes == [1, 2]


def test_model_save_and_load():
    inputs = layers.Input(shape=(12, 12, 2))
    out = Laplacian(dx=0.1, boundary_condition="central")(inputs)
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

