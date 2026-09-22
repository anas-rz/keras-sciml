from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def solve_burgers_pseudospectral_1d(
    u0: np.ndarray,
    nu: float = 0.01,
    t_end: float = 1.0,
    dt: float = 1e-3,
    length: float = 1.0,
) -> np.ndarray:
    """Solve 1D viscous Burgers' equation via pseudospectral RK4 integration.

    $$u_t + u u_x = \\nu u_{xx}$$ with periodic boundary conditions on $[0, L]$.

    Args:
        u0: ndarray of shape `(n_samples, nx)`, initial condition.
        nu: float, kinematic viscosity coefficient. Defaults to 0.01.
        t_end: float, final integration time. Defaults to 1.0.
        dt: float, time step size. Defaults to 1e-3.
        length: float, spatial domain length. Defaults to 1.0.

    Returns:
        ndarray of shape `(n_samples, nx)`, solution field at `t_end`.
    """
    n_samples, nx = u0.shape
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    # 2/3 de-aliasing mask
    k_max = (nx // 3) * (2.0 * np.pi / length)
    dealias_mask = (np.abs(k) <= k_max).astype(np.float32)

    # Linear operator in Fourier space: L = -nu * k^2
    L = -nu * (k ** 2)

    def rhs(u_hat):
        # Inverse FFT to real space
        u_real = np.fft.ifft(u_hat, axis=-1).real
        # Derivative in Fourier space: ik * u_hat
        u_x_hat = 1j * k * u_hat * dealias_mask
        u_x_real = np.fft.ifft(u_x_hat, axis=-1).real
        # Nonlinear convective term: - u * u_x
        nonlin_real = -u_real * u_x_real
        nonlin_hat = np.fft.fft(nonlin_real, axis=-1) * dealias_mask
        return L * u_hat + nonlin_hat

    u_hat = np.fft.fft(u0, axis=-1)
    n_steps = max(1, int(round(t_end / dt)))
    actual_dt = t_end / n_steps

    for _ in range(n_steps):
        # 4th-order Runge-Kutta
        k1 = rhs(u_hat)
        k2 = rhs(u_hat + 0.5 * actual_dt * k1)
        k3 = rhs(u_hat + 0.5 * actual_dt * k2)
        k4 = rhs(u_hat + actual_dt * k3)
        u_hat = u_hat + (actual_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return np.fft.ifft(u_hat, axis=-1).real.astype(np.float32)


def generate_burgers_analytical_shock(
    n_samples: int = 100,
    nx: int = 64,
    t: float = 0.5,
    nu: float = 0.05,
    c_range: Tuple[float, float] = (0.5, 1.5),
    a_range: Tuple[float, float] = (0.5, 1.0),
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate exact analytical traveling shock solutions to 1D Burgers' equation.

    $$u(x, t) = c - a \\tanh\\left(\\frac{a (x - c t - x_0)}{2\\nu}\\right)$$

    Args:
        n_samples: int, number of samples.
        nx: int, number of spatial points in [0, 1].
        t: float, evaluation time.
        nu: float, viscosity coefficient.
        c_range: Tuple[float, float], range of wave speeds.
        a_range: Tuple[float, float], range of shock amplitudes.
        seed: Optional[int], random seed.

    Returns:
        Tuple of (u0, u_t, x_grid):
        - u0: initial state at t=0, shape `(n_samples, nx, 1)`
        - u_t: solution state at time t, shape `(n_samples, nx, 1)`
        - x_grid: spatial coordinates in [0, 1], shape `(nx,)`
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)

    c = rng.uniform(c_range[0], c_range[1], (n_samples, 1)).astype(np.float32)
    a = rng.uniform(a_range[0], a_range[1], (n_samples, 1)).astype(np.float32)
    x0 = rng.uniform(0.3, 0.7, (n_samples, 1)).astype(np.float32)

    # Initial condition at t=0
    xi_0 = (a * (x[None, :] - x0)) / (2.0 * nu)
    u0 = c - a * np.tanh(xi_0)

    # Target state at t
    xi_t = (a * (x[None, :] - c * t - x0)) / (2.0 * nu)
    u_t = c - a * np.tanh(xi_t)

    return u0[:, :, None], u_t[:, :, None], x


def generate_burgers_1d(
    n_samples: int = 100,
    nx: int = 64,
    t_end: float = 1.0,
    nu: float = 0.01,
    dt: float = 1e-3,
    num_modes: int = 6,
    decay_power: float = 2.0,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 1D Burgers' equation dataset using Gaussian random field initial conditions and pseudospectral RK4.

    Initial condition $u_0(x)$ is sampled from a smooth random field and integrated forward to $t_{\\text{end}}$.

    Args:
        n_samples: int, number of samples.
        nx: int, number of grid points on [0, 1).
        t_end: float, final time. Defaults to 1.0.
        nu: float, viscosity. Defaults to 0.01.
        dt: float, time step for RK4. Defaults to 1e-3.
        num_modes: int, number of Fourier modes for $u_0$. Defaults to 6.
        decay_power: float, spectral decay factor for $u_0$. Defaults to 2.0.
        seed: Optional[int], random seed. Defaults to 42.

    Returns:
        Tuple of (u0, u_t, x_grid):
        - u0: initial condition, shape `(n_samples, nx, 1)`
        - u_t: solution at t_end, shape `(n_samples, nx, 1)`
        - x_grid: spatial coordinates on [0, 1), shape `(nx,)`
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)

    # Generate smooth initial conditions via Fourier harmonics
    k = np.arange(1, num_modes + 1, dtype=np.float32)
    scales = 1.0 / (k ** decay_power)

    a_k = rng.standard_normal((n_samples, num_modes)).astype(np.float32) * scales
    b_k = rng.standard_normal((n_samples, num_modes)).astype(np.float32) * scales

    kx = 2.0 * np.pi * k[:, None] * x[None, :]
    u0 = np.matmul(a_k, np.cos(kx)) + np.matmul(b_k, np.sin(kx))

    # Normalize amplitude
    u0 = u0 / (np.max(np.abs(u0), axis=-1, keepdims=True) + 1e-5)

    u_t = solve_burgers_pseudospectral_1d(u0, nu=nu, t_end=t_end, dt=dt, length=1.0)

    return u0[:, :, None], u_t[:, :, None], x


class BurgersDataset:
    """Convenience dataset wrapper for 1D Burgers' equation."""

    def __init__(
        self,
        mode: str = "pseudospectral",
        n_samples: int = 100,
        nx: int = 64,
        t_end: float = 1.0,
        nu: float = 0.01,
        seed: Optional[int] = 42,
    ):
        self.mode = mode.lower()
        self.n_samples = n_samples

        if self.mode == "pseudospectral":
            self.u0, self.u_t, self.grid = generate_burgers_1d(
                n_samples=n_samples, nx=nx, t_end=t_end, nu=nu, seed=seed
            )
        elif self.mode == "shock":
            self.u0, self.u_t, self.grid = generate_burgers_analytical_shock(
                n_samples=n_samples, nx=nx, t=t_end, nu=nu, seed=seed
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'pseudospectral' or 'shock'.")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.u0[idx], self.u_t[idx]

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (inputs, targets) tuple."""
        return self.u0, self.u_t

