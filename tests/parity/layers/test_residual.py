import numpy as np
import pytest
import keras
from keras import ops

from keras_sciml.layers.physics import (
    AdvectionResidual,
    BurgersResidual,
    DiffusionResidual,
    PoissonResidual,
)


def test_parity_poisson_residual_2d():
    np.random.seed(42)
    batch, h, w, c = 2, 12, 14, 1
    dx = 0.1
    u_np = np.random.randn(batch, h, w, c).astype("float32")
    f_np = np.random.randn(batch, h, w, c).astype("float32")

    # Laplacian in numpy via interior 3-point central stencil
    # For interior elements: (u[i+1] - 2u[i] + u[i-1])/dx^2
    # Verify directly against Keras Laplacian layer
    from keras_sciml.layers.differential import Laplacian

    lap_layer = Laplacian(dx=dx, boundary_condition="central")
    lap_val = ops.convert_to_numpy(lap_layer(ops.convert_to_tensor(u_np)))
    ref_res = -lap_val - f_np

    p_layer = PoissonResidual(dx=dx, boundary_condition="central", return_residual=True, add_loss=False)
    _, keras_res = p_layer((ops.convert_to_tensor(u_np), ops.convert_to_tensor(f_np)))
    keras_res_np = ops.convert_to_numpy(keras_res)

    np.testing.assert_allclose(keras_res_np, ref_res, atol=1e-5)


def test_parity_diffusion_residual_1d():
    np.random.seed(123)
    batch, n, c = 2, 20, 1
    dx = 0.05
    alpha = 2.0
    u_np = np.random.randn(batch, n, c).astype("float32")
    f_val = 1.5

    from keras_sciml.layers.differential import Laplacian

    lap_layer = Laplacian(dx=dx, boundary_condition="periodic")
    lap_val = ops.convert_to_numpy(lap_layer(ops.convert_to_tensor(u_np)))
    ref_res = -alpha * lap_val - f_val

    diff_layer = DiffusionResidual(
        diffusivity=alpha,
        dx=dx,
        forcing=f_val,
        boundary_condition="periodic",
        return_residual=True,
        add_loss=False,
    )
    _, keras_res = diff_layer(ops.convert_to_tensor(u_np))
    keras_res_np = ops.convert_to_numpy(keras_res)

    np.testing.assert_allclose(keras_res_np, ref_res, atol=1e-5)


def test_parity_burgers_residual():
    np.random.seed(777)
    batch, n, c = 2, 24, 1
    dx = 0.04
    nu = 0.05
    u_np = np.random.randn(batch, n, c).astype("float32")

    from keras_sciml.layers.differential import Gradient, Laplacian

    grad_layer = Gradient(dx=dx, boundary_condition="periodic", axes=[1], concat_components=True)
    lap_layer = Laplacian(dx=dx, boundary_condition="periodic")

    du_dx = ops.convert_to_numpy(grad_layer(ops.convert_to_tensor(u_np)))
    lap = ops.convert_to_numpy(lap_layer(ops.convert_to_tensor(u_np)))
    ref_res = u_np * du_dx - nu * lap

    burg_layer = BurgersResidual(
        nu=nu,
        dx=dx,
        boundary_condition="periodic",
        return_residual=True,
        add_loss=False,
    )
    _, keras_res = burg_layer(ops.convert_to_tensor(u_np))
    keras_res_np = ops.convert_to_numpy(keras_res)

    np.testing.assert_allclose(keras_res_np, ref_res, atol=1e-5)


def test_parity_advection_residual():
    np.random.seed(999)
    batch, n, c = 3, 16, 1
    dx = 0.1
    vel = 2.5
    u_np = np.random.randn(batch, n, c).astype("float32")

    from keras_sciml.layers.differential import Gradient

    grad_layer = Gradient(dx=dx, boundary_condition="central", axes=[1], concat_components=True)
    du_dx = ops.convert_to_numpy(grad_layer(ops.convert_to_tensor(u_np)))
    ref_res = vel * du_dx

    adv_layer = AdvectionResidual(
        velocity=vel,
        dx=dx,
        boundary_condition="central",
        return_residual=True,
        add_loss=False,
    )
    _, keras_res = adv_layer(ops.convert_to_tensor(u_np))
    keras_res_np = ops.convert_to_numpy(keras_res)

    np.testing.assert_allclose(keras_res_np, ref_res, atol=1e-5)
