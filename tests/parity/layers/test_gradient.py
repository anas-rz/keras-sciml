import numpy as np
import pytest
import keras
from keras import ops

from keras_sciml.layers.differential import Gradient


def test_parity_against_numpy_gradient_1d():
    np.random.seed(42)
    batch_size, length, channels = 3, 25, 2
    dx = 0.04

    x_np = np.random.randn(batch_size, length, channels).astype("float32")

    # Numpy reference (axis=1)
    ref_grad = np.gradient(x_np, dx, axis=1)

    layer = Gradient(dx=dx, boundary_condition="central", concat_components=True)
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(x_np)))

    np.testing.assert_allclose(keras_out, ref_grad, atol=1e-5, rtol=1e-5)


def test_parity_against_numpy_gradient_2d():
    np.random.seed(123)
    batch_size, h, w, channels = 2, 16, 20, 3
    dx = 0.1
    dy = 0.05

    x_np = np.random.randn(batch_size, h, w, channels).astype("float32")

    ref_grad_x = np.gradient(x_np, dx, axis=1)
    ref_grad_y = np.gradient(x_np, dy, axis=2)
    ref_grad = np.stack([ref_grad_x, ref_grad_y], axis=-1)

    layer = Gradient(dx=[dx, dy], boundary_condition="central", concat_components=False)
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(x_np)))

    np.testing.assert_allclose(keras_out, ref_grad, atol=1e-5, rtol=1e-5)


def test_parity_against_numpy_gradient_3d():
    np.random.seed(999)
    batch_size, d, h, w, channels = 1, 8, 10, 12, 2
    spacings = [0.2, 0.1, 0.05]

    x_np = np.random.randn(batch_size, d, h, w, channels).astype("float32")

    ref_grad_d = np.gradient(x_np, spacings[0], axis=1)
    ref_grad_h = np.gradient(x_np, spacings[1], axis=2)
    ref_grad_w = np.gradient(x_np, spacings[2], axis=3)
    ref_grad = np.stack([ref_grad_d, ref_grad_h, ref_grad_w], axis=-1)

    layer = Gradient(dx=spacings, boundary_condition="central", concat_components=False)
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(x_np)))

    np.testing.assert_allclose(keras_out, ref_grad, atol=1e-5, rtol=1e-5)


def test_parity_periodic_analytical_2d():
    # Test analytical convergence on a smooth periodic function
    n = 64
    x_coords = np.linspace(0.0, 1.0, n, endpoint=False, dtype="float32")
    y_coords = np.linspace(0.0, 1.0, n, endpoint=False, dtype="float32")
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    h = 1.0 / n

    # u = sin(2*pi*x) * cos(2*pi*y)
    u = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    u_tensor = ops.convert_to_tensor(u[None, :, :, None])

    # Exact derivatives
    exact_du_dx = 2.0 * np.pi * np.cos(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    exact_du_dy = -2.0 * np.pi * np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    exact_grad = np.stack([exact_du_dx, exact_du_dy], axis=-1)[None, :, :, :]

    layer = Gradient(dx=h, boundary_condition="periodic", concat_components=False)
    keras_out = ops.convert_to_numpy(layer(u_tensor))[:, :, :, 0, :]

    # Truncation error for second-order central difference with N=64 is < 0.05
    max_error = np.max(np.abs(keras_out - exact_grad))
    assert max_error < 0.05

