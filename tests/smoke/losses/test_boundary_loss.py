import pytest
import keras
from keras import ops
import numpy as np

from keras_sciml.losses import BoundaryLoss


def test_boundary_loss_instantiation():
    bl = BoundaryLoss(bc_type="dirichlet", value=0.0)
    assert bl.bc_type == "dirichlet"
    assert bl.value == 0.0

    bl_neumann = BoundaryLoss(bc_type="neumann", dx=0.05, sides="lower")
    assert bl_neumann.bc_type == "neumann"
    assert bl_neumann.sides == "lower"
    assert bl_neumann.dx == 0.05

    bl_robin = BoundaryLoss(bc_type="robin", alpha=2.0, beta=3.0)
    assert bl_robin.alpha == 2.0
    assert bl_robin.beta == 3.0

    bl_periodic = BoundaryLoss(bc_type="periodic")
    assert bl_periodic.bc_type == "periodic"

    with pytest.raises(ValueError, match="bc_type"):
        BoundaryLoss(bc_type="unknown")

    with pytest.raises(ValueError, match="sides"):
        BoundaryLoss(sides="invalid")

    with pytest.raises(ValueError, match="loss_type"):
        BoundaryLoss(loss_type="invalid")


def test_boundary_loss_shapes():
    # 1D field: (batch, x, channels)
    bl = BoundaryLoss(bc_type="dirichlet", value=0.0, reduction="none")
    y_pred = ops.ones((4, 20, 1))
    loss_val = bl(ops.zeros_like(y_pred), y_pred)
    assert loss_val.shape == (4,)

    # 2D field: (batch, y, x, channels)
    bl_2d = BoundaryLoss(bc_type="neumann", dx=0.1, reduction="sum_over_batch_size")
    y_pred_2d = ops.ones((2, 16, 16, 1))
    loss_val_2d = bl_2d(ops.zeros_like(y_pred_2d), y_pred_2d)
    assert loss_val_2d.shape == ()

    # Periodic
    bl_per = BoundaryLoss(bc_type="periodic", reduction="none")
    loss_val_per = bl_per(ops.zeros_like(y_pred_2d), y_pred_2d)
    assert loss_val_per.shape == (2,)

    # Collocation 1D array
    bl_colloc = BoundaryLoss(bc_type="dirichlet", value=1.0, reduction="none")
    u_b = ops.ones((10,)) * 2.0
    colloc_loss = bl_colloc(ops.zeros_like(u_b), u_b)
    assert colloc_loss.shape == (10,)
    np.testing.assert_allclose(ops.convert_to_numpy(colloc_loss), np.ones(10), rtol=1e-5)


def test_boundary_loss_serialization():
    bl = BoundaryLoss(
        bc_type="robin",
        value=1.5,
        alpha=2.0,
        beta=0.5,
        sides="lower",
        dx=0.02,
        loss_type="mae",
    )
    config = keras.saving.serialize_keras_object(bl)
    recovered = keras.saving.deserialize_keras_object(config)

    assert recovered.bc_type == "robin"
    assert recovered.value == 1.5
    assert recovered.alpha == 2.0
    assert recovered.beta == 0.5
    assert recovered.sides == "lower"
    assert recovered.dx == 0.02
    assert recovered.loss_type == "mae"

