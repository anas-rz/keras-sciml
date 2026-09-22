import numpy as np
import pytest
import scipy.ndimage as ndi
import keras
from keras import ops

from keras_sciml.layers.differential import Laplacian


def test_parity_periodic_with_scipy():
    np.random.seed(42)
    n = 16
    h = 1.0 / n
    u_np = np.random.randn(2, n, n, 2).astype("float32")

    # Reference computation using scipy.ndimage.laplace with mode='wrap'
    ref_lap = np.zeros_like(u_np)
    for b in range(2):
        for c in range(2):
            ref_lap[b, :, :, c] = ndi.laplace(u_np[b, :, :, c], mode="wrap") / (h**2)

    layer = Laplacian(dx=h, boundary_condition="periodic")
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(u_np)))

    np.testing.assert_allclose(keras_out, ref_lap, atol=1e-4, rtol=1e-4)


def test_parity_exact_polynomial_2d():
    # Exact Laplacian of 3*x^2 + 5*y^2 is 6 + 10 = 16 everywhere
    nx, ny = 12, 14
    x = np.linspace(0.0, 1.0, nx, dtype="float32")
    y = np.linspace(0.0, 1.0, ny, dtype="float32")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])

    u = (3.0 * xx**2 + 5.0 * yy**2)[None, :, :, None]  # shape (1, 12, 14, 1)

    layer = Laplacian(dx=[hx, hy], boundary_condition="central")
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(u)))

    np.testing.assert_allclose(keras_out, 16.0, atol=1e-3, rtol=1e-3)


def test_parity_analytical_periodic_eigenmode():
    n = 64
    coords = np.linspace(0.0, 1.0, n, endpoint=False, dtype="float32")
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    h = 1.0 / n

    # u = sin(2*pi*x) * cos(2*pi*y)
    # Delta u = -( (2*pi)^2 + (2*pi)^2 ) * u = -8*pi^2 * u
    u = (np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy))[None, :, :, None]
    exact_lap = -8.0 * (np.pi**2) * u

    layer = Laplacian(dx=h, boundary_condition="periodic")
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(u)))

    # Second-order truncation error for N=64 on smooth sinusoid is < 0.25 (relative error < 0.3%)
    max_error = np.max(np.abs(keras_out - exact_lap))
    assert max_error < 0.25

