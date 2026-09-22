import pytest
import keras
from keras import ops
import numpy as np

from keras_sciml.losses import ConservationLoss


def test_conservation_loss_instantiation():
    cl = ConservationLoss()
    assert cl.reduction_type == "integral"
    assert cl.target is None
    assert cl.dx == 1.0

    cl_sum = ConservationLoss(reduction_type="sum", target=10.0, relative=True)
    assert cl_sum.reduction_type == "sum"
    assert cl_sum.target == 10.0
    assert cl_sum.relative

    cl_div = ConservationLoss(reduction_type="incompressibility")
    assert cl_div.reduction_type == "incompressibility"

    with pytest.raises(ValueError, match="reduction_type"):
        ConservationLoss(reduction_type="invalid")

    with pytest.raises(ValueError, match="loss_type"):
        ConservationLoss(loss_type="invalid")


def test_conservation_loss_shapes():
    # 1D field
    cl_1d = ConservationLoss(reduction_type="integral", target=1.0, dx=0.1, reduction="none")
    y_pred = ops.ones((4, 20, 1))
    loss_1d = cl_1d(ops.zeros_like(y_pred), y_pred)
    assert loss_1d.shape == (4,)

    # 2D field comparison with y_true
    cl_2d = ConservationLoss(reduction_type="mean", reduction="sum_over_batch_size")
    y_true = ops.ones((2, 10, 10, 1))
    y_pred = ops.ones((2, 10, 10, 1)) * 2.0
    loss_2d = cl_2d(y_true, y_pred)
    assert loss_2d.shape == ()

    # Incompressibility / divergence on vector field
    cl_div = ConservationLoss(reduction_type="incompressibility", reduction="none")
    # 2D velocity vector field (u, v): (batch, y, x, 2)
    y_vec = ops.ones((3, 8, 8, 2))
    loss_div = cl_div(ops.zeros_like(y_vec), y_vec)
    assert loss_div.shape == (3,)


def test_conservation_loss_serialization():
    cl = ConservationLoss(
        reduction_type="integral",
        target=5.0,
        dx=[0.1, 0.2],
        axes=(1, 2),
        relative=True,
        loss_type="mae",
    )
    config = keras.saving.serialize_keras_object(cl)
    recovered = keras.saving.deserialize_keras_object(config)

    assert recovered.reduction_type == "integral"
    assert recovered.target == 5.0
    assert recovered.dx == [0.1, 0.2]
    assert recovered.axes == (1, 2)
    assert recovered.relative
    assert recovered.loss_type == "mae"

