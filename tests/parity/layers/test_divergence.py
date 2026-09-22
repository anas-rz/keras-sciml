import numpy as np
import pytest
import keras
from keras import ops

from keras_sciml.layers.differential import Divergence


def test_parity_against_numpy_gradient_1d():
    np.random.seed(42)
    batch_size, length = 3, 20
    dx = 0.05
    v_np = np.random.randn(batch_size, length, 1).astype("float32")

    ref_div = np.gradient(v_np[:, :, 0], dx, axis=1)[..., None]

    layer = Divergence(dx=dx, boundary_condition="central", keepdims=True)
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    np.testing.assert_allclose(keras_out, ref_div, atol=1e-5, rtol=1e-5)


def test_parity_against_numpy_gradient_2d():
    np.random.seed(123)
    batch_size, h, w = 2, 14, 16
    dx = 0.1
    dy = 0.08
    v_np = np.random.randn(batch_size, h, w, 2).astype("float32")

    ref_dvx_dx = np.gradient(v_np[:, :, :, 0], dx, axis=1)
    ref_dvy_dy = np.gradient(v_np[:, :, :, 1], dy, axis=2)
    ref_div = (ref_dvx_dx + ref_dvy_dy)[..., None]

    layer = Divergence(dx=[dx, dy], boundary_condition="central", keepdims=True)
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    np.testing.assert_allclose(keras_out, ref_div, atol=1e-5, rtol=1e-5)


def test_parity_against_numpy_gradient_3d():
    np.random.seed(999)
    batch_size, d, h, w = 1, 8, 10, 12
    spacings = [0.2, 0.1, 0.05]
    v_np = np.random.randn(batch_size, d, h, w, 3).astype("float32")

    ref_dvx_dx = np.gradient(v_np[:, :, :, :, 0], spacings[0], axis=1)
    ref_dvy_dy = np.gradient(v_np[:, :, :, :, 1], spacings[1], axis=2)
    ref_dvz_dz = np.gradient(v_np[:, :, :, :, 2], spacings[2], axis=3)
    ref_div = (ref_dvx_dx + ref_dvy_dy + ref_dvz_dz)[..., None]

    layer = Divergence(dx=spacings, boundary_condition="central", keepdims=True)
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    np.testing.assert_allclose(keras_out, ref_div, atol=1e-5, rtol=1e-5)


def test_parity_solenoidal_divergence_free():
    # v = (-sin(2*pi*y), sin(2*pi*x)) -> div(v) = d/dx(-sin(2*pi*y)) + d/dy(sin(2*pi*x)) = 0 + 0 = 0
    n = 32
    coords = np.linspace(0.0, 1.0, n, endpoint=False, dtype="float32")
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    h = 1.0 / n

    vx = -np.sin(2.0 * np.pi * yy)
    vy = np.sin(2.0 * np.pi * xx)
    v_np = np.stack([vx, vy], axis=-1)[None, ...]

    layer = Divergence(dx=h, boundary_condition="periodic")
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    # Exact zero to single precision roundoff
    np.testing.assert_allclose(keras_out, 0.0, atol=1e-6)


def test_parity_analytical_periodic():
    # v = (sin(2*pi*x)*cos(2*pi*y), cos(2*pi*x)*sin(2*pi*y))
    # div(v) = 2*pi*cos(2*pi*x)*cos(2*pi*y) + 2*pi*cos(2*pi*x)*cos(2*pi*y) = 4*pi*cos(2*pi*x)*cos(2*pi*y)
    n = 64
    coords = np.linspace(0.0, 1.0, n, endpoint=False, dtype="float32")
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    h = 1.0 / n

    vx = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    vy = np.cos(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    v_np = np.stack([vx, vy], axis=-1)[None, ...]
    exact_div = (4.0 * np.pi * np.cos(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy))[None, ..., None]

    layer = Divergence(dx=h, boundary_condition="periodic")
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    max_error = np.max(np.abs(keras_out - exact_div))
    assert max_error < 0.05

