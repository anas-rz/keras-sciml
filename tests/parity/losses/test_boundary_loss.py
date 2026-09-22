import numpy as np
import pytest
from keras import ops

from keras_sciml.losses import BoundaryLoss


def test_dirichlet_boundary_parity():
    rng = np.random.default_rng(42)
    # 2D field: (batch=3, ny=10, nx=10, channels=1)
    y_pred = rng.standard_normal((3, 10, 10, 1)).astype(np.float32)

    # Both sides (axes=(1, 2)) with value=0.0
    bl = BoundaryLoss(bc_type="dirichlet", value=0.0, reduction="none")

    # Manual reference:
    # axis 1: y=0 (y_pred[:, 0, :, :]), y=ny-1 (y_pred[:, -1, :, :])
    # axis 2: x=0 (y_pred[:, :, 0, :]), x=nx-1 (y_pred[:, :, -1, :])
    r1_low = np.mean(y_pred[:, 0, :, :] ** 2, axis=(1, 2))
    r1_high = np.mean(y_pred[:, -1, :, :] ** 2, axis=(1, 2))
    r2_low = np.mean(y_pred[:, :, 0, :] ** 2, axis=(1, 2))
    r2_high = np.mean(y_pred[:, :, -1, :] ** 2, axis=(1, 2))
    ref_loss = r1_low + r1_high + r2_low + r2_high

    keras_loss = ops.convert_to_numpy(bl(ops.zeros_like(y_pred), y_pred))
    np.testing.assert_allclose(keras_loss, ref_loss, rtol=1e-5, atol=1e-5)


def test_neumann_boundary_parity():
    rng = np.random.default_rng(77)
    # 1D field: (batch=2, nx=20, channels=1)
    y_pred = rng.standard_normal((2, 20, 1)).astype(np.float32)
    dx = 0.05

    bl = BoundaryLoss(bc_type="neumann", value=0.0, dx=dx, reduction="none")

    # Lower: (u_0 - u_1) / dx
    # Upper: (u_{-1} - u_{-2}) / dx
    du_dn_low = (y_pred[:, 0, :] - y_pred[:, 1, :]) / dx
    du_dn_high = (y_pred[:, -1, :] - y_pred[:, -2, :]) / dx

    ref_loss = np.mean(du_dn_low ** 2, axis=1) + np.mean(du_dn_high ** 2, axis=1)
    keras_loss = ops.convert_to_numpy(bl(ops.zeros_like(y_pred), y_pred))
    np.testing.assert_allclose(keras_loss, ref_loss, rtol=1e-5, atol=1e-5)


def test_periodic_boundary_parity():
    rng = np.random.default_rng(88)
    y_pred = rng.standard_normal((4, 15, 1)).astype(np.float32)

    bl = BoundaryLoss(bc_type="periodic", reduction="none")
    ref_loss = np.mean((y_pred[:, 0, :] - y_pred[:, -1, :]) ** 2, axis=1)

    keras_loss = ops.convert_to_numpy(bl(ops.zeros_like(y_pred), y_pred))
    np.testing.assert_allclose(keras_loss, ref_loss, rtol=1e-5, atol=1e-5)

