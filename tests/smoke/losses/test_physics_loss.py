import pytest
import keras
from keras import ops
import numpy as np

from keras_sciml.losses import (
    H1Loss,
    PhysicsLoss,
    RelativeL2Loss,
    relative_l2_error,
)


def test_relative_l2_instantiation():
    loss = RelativeL2Loss()
    assert loss.p == 2.0
    assert not loss.squared
    assert loss.epsilon == 1e-7

    loss_sq = RelativeL2Loss(p=1.0, squared=True, epsilon=1e-5, axes=(1, 2))
    assert loss_sq.p == 1.0
    assert loss_sq.squared
    assert loss_sq.epsilon == 1e-5
    assert loss_sq.axes == (1, 2)


def test_relative_l2_shapes_and_values():
    loss_fn = RelativeL2Loss(reduction="none")
    y_true = ops.ones((4, 16, 16, 2))
    y_pred = ops.ones((4, 16, 16, 2)) * 2.0

    res = loss_fn(y_true, y_pred)
    assert res.shape == (4,)
    # ||2 - 1||_2 / ||1||_2 = 1.0
    np.testing.assert_allclose(ops.convert_to_numpy(res), np.ones(4), rtol=1e-5)

    # Reduction sum_over_batch_size
    loss_reduced = RelativeL2Loss(reduction="sum_over_batch_size")
    scalar_res = loss_reduced(y_true, y_pred)
    assert scalar_res.shape == ()
    np.testing.assert_allclose(ops.convert_to_numpy(scalar_res), 1.0, rtol=1e-5)

    # 1D unbatched / vector shapes
    y_true_1d = ops.ones((8,))
    y_pred_1d = ops.ones((8,)) * 3.0
    loss_1d = RelativeL2Loss(reduction="none")
    res_1d = loss_1d(y_true_1d, y_pred_1d)
    assert res_1d.shape == (8,)

    # Functional form
    func_res = relative_l2_error(y_true, y_pred)
    np.testing.assert_allclose(ops.convert_to_numpy(func_res), np.ones(4), rtol=1e-5)


def test_relative_l2_serialization():
    loss = RelativeL2Loss(p=1.5, squared=True, epsilon=1e-6, axes=(1, 2))
    config = keras.saving.serialize_keras_object(loss)
    recovered = keras.saving.deserialize_keras_object(config)

    assert recovered.p == 1.5
    assert recovered.squared
    assert recovered.epsilon == 1e-6
    assert recovered.axes == (1, 2)


def test_h1_loss_instantiation():
    h1 = H1Loss(alpha=0.5, dx=0.01)
    assert h1.alpha == 0.5
    assert h1.dx == 0.01
    assert not h1.relative

    h1_rel = H1Loss(relative=True, axes=(1, 2))
    assert h1_rel.relative
    assert h1_rel.axes == (1, 2)


def test_h1_loss_shapes():
    h1 = H1Loss(reduction="none")
    y_true = ops.zeros((3, 20, 1))
    y_pred = ops.ones((3, 20, 1))
    res = h1(y_true, y_pred)
    assert res.shape == (3,)

    # 2D field
    h1_2d = H1Loss(relative=True, reduction="sum_over_batch_size")
    y_true_2d = ops.ones((2, 10, 10, 1))
    y_pred_2d = ops.ones((2, 10, 10, 1)) * 1.5
    res_2d = h1_2d(y_true_2d, y_pred_2d)
    assert res_2d.shape == ()


def test_h1_loss_serialization():
    h1 = H1Loss(alpha=0.2, dx=0.05, relative=True)
    config = keras.saving.serialize_keras_object(h1)
    recovered = keras.saving.deserialize_keras_object(config)

    assert recovered.alpha == 0.2
    assert recovered.dx == 0.05
    assert recovered.relative


def test_physics_loss_instantiation_and_shapes():
    pl_rel = PhysicsLoss(loss_type="relative_l2")
    assert pl_rel.loss_type == "relative_l2"

    pl_h1 = PhysicsLoss(loss_type="h1", alpha=0.05)
    assert pl_h1.alpha == 0.05

    pl_mse = PhysicsLoss(loss_type="mse")
    assert pl_mse.loss_type == "mse"

    with pytest.raises(ValueError, match="loss_type"):
        PhysicsLoss(loss_type="unsupported")

    y_true = ops.ones((2, 12, 1))
    y_pred = ops.ones((2, 12, 1)) * 2.0
    val = pl_rel(y_true, y_pred)
    assert val.shape == ()

    # With residual function
    def dummy_pde(u):
        return u - 1.0

    pl_with_pde = PhysicsLoss(loss_type="mse", pde_weight=0.5, residual_fn=dummy_pde)
    loss_val = pl_with_pde(y_true, y_pred)
    assert loss_val.shape == ()


def test_physics_loss_serialization():
    pl = PhysicsLoss(loss_type="relative_l2", data_weight=2.0, pde_weight=0.1, alpha=0.3)
    config = keras.saving.serialize_keras_object(pl)
    recovered = keras.saving.deserialize_keras_object(config)

    assert recovered.loss_type == "relative_l2"
    assert recovered.data_weight == 2.0
    assert recovered.pde_weight == 0.1
    assert recovered.alpha == 0.3

