import numpy as np
import pytest

from keras_sciml.datasets import generate_darcy_2d


def test_darcy_2d_residual_parity():
    # Verify that -nabla cdot (a nabla u) = f on the interior of Darcy solutions
    res = 16
    f_val = 1.0
    a, u, x, y = generate_darcy_2d(n_samples=2, resolution=res, f_val=f_val, seed=77)
    h = 1.0 / (res - 1)

    a_grid = a[0, :, :, 0]
    u_grid = u[0, :, :, 0]

    # Compute discrete divergence: -div(a grad u) for interior points
    # a_{i+1/2, j} * (u_{i+1, j} - u_{i, j}) / h - a_{i-1/2, j} * (u_{i, j} - u_{i-1, j}) / h
    # Interface conductivities: arithmetic mean
    flux_x = 0.5 * (a_grid[:, 1:] + a_grid[:, :-1]) * (u_grid[:, 1:] - u_grid[:, :-1]) / h
    flux_y = 0.5 * (a_grid[1:, :] + a_grid[:-1, :]) * (u_grid[1:, :] - u_grid[:-1, :]) / h

    # Divergence at cell (i, j) for i in [1, res-2], j in [1, res-2]
    div_flux_x = (flux_x[1:-1, 1:] - flux_x[1:-1, :-1]) / h
    div_flux_y = (flux_y[1:, 1:-1] - flux_y[:-1, 1:-1]) / h

    div_total = -(div_flux_x + div_flux_y)

    # Compare with f_val
    residual = div_total - f_val
    # Max error on CG solution with tol=1e-6
    assert np.max(np.abs(residual)) < 1e-3

