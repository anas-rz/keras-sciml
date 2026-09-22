import numpy as np
import pytest

from keras_sciml.datasets import (
    generate_burgers_1d,
    generate_burgers_analytical_shock,
)


def test_burgers_analytical_shock_parity():
    # Verify that analytical traveling shock satisfies Burgers PDE: u_t + u u_x - nu u_{xx} = 0
    nx = 64
    nu = 0.05
    t0 = 0.4
    dt = 1e-4

    _, u_minus, x = generate_burgers_analytical_shock(n_samples=2, nx=nx, t=t0 - dt, nu=nu, seed=42)
    _, u_plus, _ = generate_burgers_analytical_shock(n_samples=2, nx=nx, t=t0 + dt, nu=nu, seed=42)
    _, u_curr, _ = generate_burgers_analytical_shock(n_samples=2, nx=nx, t=t0, nu=nu, seed=42)

    u = u_curr[:, :, 0]
    h = x[1] - x[0]

    # Time derivative: u_t
    u_t = (u_plus[:, :, 0] - u_minus[:, :, 0]) / (2.0 * dt)

    # Spatial derivatives (interior)
    u_x = (u[:, 2:] - u[:, :-2]) / (2.0 * h)
    u_xx = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (h ** 2)

    # Residual: u_t + u u_x - nu u_xx
    residual = u_t[:, 1:-1] + u[:, 1:-1] * u_x - nu * u_xx

    # Check that residual is near zero
    assert np.mean(np.abs(residual)) < 0.02
    assert np.max(np.abs(residual)) < 0.1


def test_burgers_energy_dissipation_parity():
    # In viscous Burgers equation, kinetic energy E(t) = 0.5 * int u^2 dx is strictly dissipated: dE/dt <= 0
    u0, ut, x = generate_burgers_1d(n_samples=3, nx=32, t_end=0.5, nu=0.02, dt=1e-3, seed=99)
    e0 = np.sum(u0[:, :, 0] ** 2, axis=1)
    et = np.sum(ut[:, :, 0] ** 2, axis=1)

    # Energy must decrease or remain constant
    assert np.all(et < e0)
