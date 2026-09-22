import numpy as np
import pytest
import keras
from keras import ops

from keras_sciml.layers.physics import (
    BoundaryCondition,
    DirichletBC,
    NeumannBC,
    PeriodicBC,
    RobinBC,
)


def test_parity_dirichlet_residuals_2d():
    np.random.seed(42)
    batch, h, w, c = 2, 12, 14, 1
    u_np = np.random.randn(batch, h, w, c).astype("float32")
    g = 3.5

    # Target residuals in numpy:
    # Axis 1 (h):
    r_ax1_low = u_np[:, 0:1, :, :] - g
    r_ax1_high = u_np[:, -1:, :, :] - g
    # Axis 2 (w):
    r_ax2_low = u_np[:, :, 0:1, :] - g
    r_ax2_high = u_np[:, :, -1:, :] - g

    ref_res = np.concatenate(
        [
            r_ax1_low.reshape(-1),
            r_ax1_high.reshape(-1),
            r_ax2_low.reshape(-1),
            r_ax2_high.reshape(-1),
        ],
        axis=0,
    )

    bc = DirichletBC(value=g, return_residual=True, add_loss=False)
    _, keras_res = bc(ops.convert_to_tensor(u_np))
    keras_res_np = ops.convert_to_numpy(keras_res)

    np.testing.assert_allclose(keras_res_np, ref_res, atol=1e-6)


def test_parity_neumann_residuals_1d():
    np.random.seed(123)
    batch, n, c = 2, 20, 1
    dx = 0.05
    u_np = np.random.randn(batch, n, c).astype("float32")
    g = 0.5

    # Outward normal derivatives:
    # lower (at 0): normal is -x -> (u[0] - u[1]) / dx
    r_low = (u_np[:, 0:1, :] - u_np[:, 1:2, :]) / dx - g
    # upper (at -1): normal is +x -> (u[-1] - u[-2]) / dx
    r_high = (u_np[:, -1:, :] - u_np[:, -2:-1, :]) / dx - g

    ref_res = np.concatenate([r_low.reshape(-1), r_high.reshape(-1)], axis=0)

    bc = NeumannBC(value=g, dx=dx, return_residual=True, add_loss=False)
    _, keras_res = bc(ops.convert_to_tensor(u_np))
    keras_res_np = ops.convert_to_numpy(keras_res)

    np.testing.assert_allclose(keras_res_np, ref_res, atol=1e-5)


def test_parity_robin_residuals():
    np.random.seed(999)
    batch, n, c = 2, 16, 1
    dx = 0.1
    alpha, beta = 2.0, 0.5
    g = 1.2
    u_np = np.random.randn(batch, n, c).astype("float32")

    # Lower: alpha * u_0 + beta * (u_0 - u_1) / dx - g
    du_dn_low = (u_np[:, 0:1, :] - u_np[:, 1:2, :]) / dx
    r_low = alpha * u_np[:, 0:1, :] + beta * du_dn_low - g

    # Upper: alpha * u_{-1} + beta * (u_{-1} - u_{-2}) / dx - g
    du_dn_high = (u_np[:, -1:, :] - u_np[:, -2:-1, :]) / dx
    r_high = alpha * u_np[:, -1:, :] + beta * du_dn_high - g

    ref_res = np.concatenate([r_low.reshape(-1), r_high.reshape(-1)], axis=0)

    bc = RobinBC(alpha=alpha, beta=beta, value=g, dx=dx, return_residual=True, add_loss=False)
    _, keras_res = bc(ops.convert_to_tensor(u_np))
    keras_res_np = ops.convert_to_numpy(keras_res)

    np.testing.assert_allclose(keras_res_np, ref_res, atol=1e-5)


def test_parity_periodic_residuals():
    np.random.seed(777)
    batch, n, c = 3, 24, 2
    u_np = np.random.randn(batch, n, c).astype("float32")

    ref_res = u_np[:, 0:1, :] - u_np[:, -1:, :]

    bc = PeriodicBC(return_residual=True, add_loss=False)
    _, keras_res = bc(ops.convert_to_tensor(u_np))
    keras_res_np = ops.convert_to_numpy(keras_res)

    np.testing.assert_allclose(keras_res_np, ref_res, atol=1e-6)


def test_parity_hard_dirichlet_2d():
    batch, h, w, c = 2, 8, 10, 1
    u_np = np.ones((batch, h, w, c), dtype="float32") * 10.0
    g = -1.0

    ref_out = u_np.copy()
    ref_out[:, 0, :, :] = g
    ref_out[:, -1, :, :] = g
    ref_out[:, :, 0, :] = g
    ref_out[:, :, -1, :] = g

    bc = DirichletBC(value=g, enforcement="hard")
    keras_out = ops.convert_to_numpy(bc(ops.convert_to_tensor(u_np)))

    np.testing.assert_allclose(keras_out, ref_out, atol=1e-6)
