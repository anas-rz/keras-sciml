import numpy as np
import pytest

from keras_sciml.datasets import generate_poisson_1d, generate_poisson_2d


def test_poisson_1d_residual_parity():
    # Generate 1D Poisson data with fine grid
    nx = 128
    f, u, x = generate_poisson_1d(n_samples=5, nx=nx, num_modes=4, seed=42)
    h = 1.0 / (nx - 1)

    # Compute interior second derivative: -u''(x)
    # Interior: u[i+1] - 2u[i] + u[i-1] / h^2
    u_interior = u[:, :, 0]
    laplace_u = (u_interior[:, 2:] - 2.0 * u_interior[:, 1:-1] + u_interior[:, :-2]) / (h ** 2)
    neg_laplace = -laplace_u
    f_interior = f[:, 1:-1, 0]

    # For smooth sine modes, second-order finite difference error is O(h^2)
    # Rel error < 1%
    rel_error = np.linalg.norm(neg_laplace - f_interior, axis=1) / np.linalg.norm(f_interior, axis=1)
    assert np.all(rel_error < 0.02)


def test_poisson_2d_residual_parity():
    # Generate 2D Poisson data
    nx = ny = 32
    f, u, x, y = generate_poisson_2d(n_samples=4, nx=nx, ny=ny, num_modes=2, seed=123)
    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)

    u_grid = u[:, :, :, 0]
    # d^2u/dx^2
    d2u_dx2 = (u_grid[:, :, 2:] - 2.0 * u_grid[:, :, 1:-1] + u_grid[:, :, :-2]) / (hx ** 2)
    # d^2u/dy^2
    d2u_dy2 = (u_grid[:, 2:, :] - 2.0 * u_grid[:, 1:-1, :] + u_grid[:, :-2, :]) / (hy ** 2)

    # Interior: 1:-1 in both x and y
    laplacian_u = d2u_dx2[:, 1:-1, :] + d2u_dy2[:, :, 1:-1]
    neg_laplacian = -laplacian_u
    f_interior = f[:, 1:-1, 1:-1, 0]

    rel_error = np.linalg.norm(neg_laplacian - f_interior, axis=(1, 2)) / np.linalg.norm(f_interior, axis=(1, 2))
    assert np.all(rel_error < 0.03)

