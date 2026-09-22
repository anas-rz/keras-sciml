import tempfile
import pytest
import keras
from keras import layers, ops
import numpy as np

from keras_sciml.layers.operators import (
    DeepONet,
    DeepONetCartesianProduct,
    DeepONetPointwise,
)


def test_instantiation():
    cart = DeepONetCartesianProduct(use_bias=True)
    assert cart.use_bias is True

    pt = DeepONetPointwise(use_bias=False)
    assert pt.use_bias is False

    unified = DeepONet(mode="cartesian")
    assert unified.mode == "cartesian"

    with pytest.raises(ValueError, match="must be 'cartesian' or 'pointwise'"):
        DeepONet(mode="invalid_mode")


def test_dimension_mismatch_error():
    cart = DeepONetCartesianProduct()
    b = ops.ones((4, 16))
    t = ops.ones((10, 20))  # 20 != 16
    with pytest.raises(ValueError, match="dimension must match"):
        cart([b, t])


def test_cartesian_shapes_unbatched_trunk():
    cart = DeepONetCartesianProduct(use_bias=True)
    b = ops.ones((4, 16))  # 4 functions, latent p=16
    t = ops.ones((10, 16))  # 10 points, latent p=16
    out = cart([b, t])
    assert out.shape == (4, 10)
    assert cart.compute_output_shape([(4, 16), (10, 16)]) == (4, 10)


def test_cartesian_shapes_batched_trunk():
    cart = DeepONetCartesianProduct(use_bias=False)
    b = ops.ones((4, 16))  # 4 functions
    t = ops.ones((4, 10, 16))  # 4 batches, 10 points per function
    out = cart([b, t])
    assert out.shape == (4, 10)
    assert cart.compute_output_shape([(4, 16), (4, 10, 16)]) == (4, 10)


def test_pointwise_shapes_2d():
    pt = DeepONetPointwise(use_bias=True)
    b = ops.ones((8, 16))
    t = ops.ones((8, 16))
    out = pt([b, t])
    assert out.shape == (8, 1)
    assert pt.compute_output_shape([(8, 16), (8, 16)]) == (8, 1)


def test_pointwise_shapes_3d():
    pt = DeepONetPointwise(use_bias=False)
    b = ops.ones((4, 12, 16))
    t = ops.ones((4, 12, 16))
    out = pt([b, t])
    assert out.shape == (4, 12, 1)
    assert pt.compute_output_shape([(4, 12, 16), (4, 12, 16)]) == (4, 12, 1)


def test_unified_deeponet_with_subnetworks():
    branch_net = layers.Dense(16, activation="relu")
    trunk_net = layers.Dense(16, activation="relu")
    deeponet = DeepONet(branch_net=branch_net, trunk_net=trunk_net, mode="cartesian")

    u = ops.ones((4, 30))  # 30 sensors
    y = ops.ones((10, 2))  # 2D coordinates, 10 points
    out = deeponet([u, y])
    assert out.shape == (4, 10)


def test_serialization_config():
    cart = DeepONetCartesianProduct(use_bias=True)
    cfg = cart.get_config()
    rec = DeepONetCartesianProduct.from_config(cfg)
    assert rec.use_bias is True

    pt = DeepONetPointwise(use_bias=False)
    cfg_pt = pt.get_config()
    rec_pt = DeepONetPointwise.from_config(cfg_pt)
    assert rec_pt.use_bias is False


def test_model_save_and_load_roundtrip():
    u_in = layers.Input(shape=(20,))
    y_in = layers.Input(shape=(10, 16))
    b = layers.Dense(16)(u_in)
    out = DeepONetCartesianProduct(use_bias=True)([b, y_in])

    model = keras.Model(inputs=[u_in, y_in], outputs=out)
    test_u = np.random.uniform(size=(3, 20)).astype("float32")
    test_y = np.random.uniform(size=(3, 10, 16)).astype("float32")
    orig_pred = model([test_u, test_y])

    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        model.save(tmp.name)
        loaded = keras.models.load_model(tmp.name)
        loaded_pred = loaded([test_u, test_y])

    np.testing.assert_allclose(
        ops.convert_to_numpy(orig_pred),
        ops.convert_to_numpy(loaded_pred),
        atol=1e-5,
    )

