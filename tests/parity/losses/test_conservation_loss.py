import numpy as np
import pytest
from keras import ops

from keras_sciml.losses import ConservationLoss


def test_conservation_integral_parity():
    rng = np.random.default_rng(42)
    # 2D field: (batch=3, ny=8, nx=8, channels=1)
    y_pred = rng.standard_normal((3, 8, 8, 1)).astype(np.float32)
    dx = [0.1, 0.2]
    target = 2.5

    cl = ConservationLoss(reduction_type="integral", target=target, dx=dx, reduction="none")

    # Volume element = 0.1 * 0.2 = 0.02
    cell_vol = 0.1 * 0.2
    integral_np = np.sum(y_pred, axis=(1, 2)) * cell_vol
    ref_loss = np.mean((integral_np - target) ** 2, axis=-1)

    keras_loss = ops.convert_to_numpy(cl(ops.zeros_like(y_pred), y_pred))
    np.testing.assert_allclose(keras_loss, ref_loss, rtol=1e-5, atol=1e-5)


def test_conservation_mean_parity():
    rng = np.random.default_rng(55)
    y_pred = rng.standard_normal((4, 12, 1)).astype(np.float32)
    target = 0.0

    cl = ConservationLoss(reduction_type="mean", target=target, reduction="none")
    mean_np = np.mean(y_pred, axis=1)
    ref_loss = np.mean((mean_np - target) ** 2, axis=-1)

    keras_loss = ops.convert_to_numpy(cl(ops.zeros_like(y_pred), y_pred))
    np.testing.assert_allclose(keras_loss, ref_loss, rtol=1e-5, atol=1e-5)


def test_conservation_sum_relative_parity():
    rng = np.random.default_rng(66)
    y_true = np.abs(rng.standard_normal((3, 10, 1)).astype(np.float32)) + 0.5
    y_pred = y_true * 1.1

    cl = ConservationLoss(reduction_type="sum", relative=True, reduction="none")
    sum_pred = np.sum(y_pred, axis=1)
    sum_true = np.sum(y_true, axis=1)
    ref_loss = np.mean(((sum_pred - sum_true) / (np.abs(sum_true) + 1e-7)) ** 2, axis=-1)

    keras_loss = ops.convert_to_numpy(cl(y_true, y_pred))
    np.testing.assert_allclose(keras_loss, ref_loss, rtol=1e-4, atol=1e-5)

