from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def generate_poisson_1d(
    n_samples: int = 100,
    nx: int = 64,
    num_modes: int = 8,
    decay_power: float = 2.0,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 1D Poisson equation dataset: -u''(x) = f(x) with u(0)=u(1)=0.

    Uses analytical sine series decomposition:
    $$f(x) = \\sum_{k=1}^K c_k \\sin(k \\pi x), \\quad u(x) = \\sum_{k=1}^K \\frac{c_k}{(k \\pi)^2} \\sin(k \\pi x)$$

    Args:
        n_samples: int, number of samples to generate. Defaults to 100.
        nx: int, number of spatial grid points in [0, 1]. Defaults to 64.
        num_modes: int, number of Fourier sine modes. Defaults to 8.
        decay_power: float, spectral decay factor for coefficients $c_k \\propto k^{-\\text{decay}}$. Defaults to 2.0.
        seed: Optional[int], random seed. Defaults to 42.

    Returns:
        Tuple of (forcing, solution, grid):
        - forcing: ndarray of shape `(n_samples, nx, 1)`
        - solution: ndarray of shape `(n_samples, nx, 1)`
        - grid: ndarray of shape `(nx,)`
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)

    # Random mode amplitudes: c_k ~ N(0, 1) * k^(-decay)
    k = np.arange(1, num_modes + 1, dtype=np.float32)
    scales = 1.0 / (k ** decay_power)

    # (n_samples, num_modes)
    coeffs = rng.standard_normal((n_samples, num_modes)).astype(np.float32) * scales

    # Basis functions: sin(k * pi * x), shape: (num_modes, nx)
    kx = k[:, None] * np.pi * x[None, :]
    basis = np.sin(kx)  # (num_modes, nx)

    # Forcing: sum c_k * sin(k * pi * x)
    f = np.matmul(coeffs, basis)[:, :, None]  # (n_samples, nx, 1)

    # Solution: sum c_k / (k * pi)^2 * sin(k * pi * x)
    sol_basis = basis / ((k[:, None] * np.pi) ** 2)
    u = np.matmul(coeffs, sol_basis)[:, :, None]  # (n_samples, nx, 1)

    return f, u, x


def generate_poisson_2d(
    n_samples: int = 100,
    nx: int = 32,
    ny: int = 32,
    num_modes: int = 4,
    decay_power: float = 2.0,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 2D Poisson equation dataset: -Delta u = f with u = 0 on boundary.

    Uses 2D analytical sine series decomposition:
    $$f(x, y) = \\sum_{k_x, k_y} c_{k_x, k_y} \\sin(k_x \\pi x) \\sin(k_y \\pi y)$$
    $$u(x, y) = \\sum_{k_x, k_y} \\frac{c_{k_x, k_y}}{\\pi^2 (k_x^2 + k_y^2)} \\sin(k_x \\pi x) \\sin(k_y \\pi y)$$

    Args:
        n_samples: int, number of samples to generate. Defaults to 100.
        nx: int, number of grid points along x in [0, 1]. Defaults to 32.
        ny: int, number of grid points along y in [0, 1]. Defaults to 32.
        num_modes: int, number of Fourier modes along each axis. Defaults to 4.
        decay_power: float, spectral decay factor. Defaults to 2.0.
        seed: Optional[int], random seed. Defaults to 42.

    Returns:
        Tuple of (forcing, solution, x_grid, y_grid):
        - forcing: ndarray of shape `(n_samples, ny, nx, 1)`
        - solution: ndarray of shape `(n_samples, ny, nx, 1)`
        - x_grid: ndarray of shape `(nx,)`
        - y_grid: ndarray of shape `(ny,)`
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)

    kx = np.arange(1, num_modes + 1, dtype=np.float32)
    ky = np.arange(1, num_modes + 1, dtype=np.float32)

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    KX_flat = KX.ravel()
    KY_flat = KY.ravel()
    total_modes = len(KX_flat)

    k_norm_sq = KX_flat ** 2 + KY_flat ** 2
    scales = 1.0 / (k_norm_sq ** (decay_power / 2.0))

    coeffs = rng.standard_normal((n_samples, total_modes)).astype(np.float32) * scales

    # Basis functions: sin(kx * pi * x) * sin(ky * pi * y)
    # Shape: (total_modes, ny, nx)
    basis = np.sin(KX_flat[:, None, None] * np.pi * x[None, None, :]) * np.sin(
        KY_flat[:, None, None] * np.pi * y[None, :, None]
    )

    basis_flat = basis.reshape(total_modes, ny * nx)  # (total_modes, ny*nx)
    f_flat = np.matmul(coeffs, basis_flat)  # (n_samples, ny*nx)
    f = f_flat.reshape(n_samples, ny, nx, 1)

    sol_basis_flat = (basis_flat / ((k_norm_sq[:, None] * (np.pi ** 2))))
    u_flat = np.matmul(coeffs, sol_basis_flat)
    u = u_flat.reshape(n_samples, ny, nx, 1)

    return f, u, x, y


class PoissonDataset:
    """Convenience dataset wrapper for 1D and 2D Poisson equations."""

    def __init__(
        self,
        dim: int = 1,
        n_samples: int = 100,
        resolution: Union[int, Tuple[int, int]] = 64,
        num_modes: int = 6,
        seed: Optional[int] = 42,
    ):
        self.dim = dim
        self.n_samples = n_samples
        self.seed = seed

        if dim == 1:
            nx = resolution if isinstance(resolution, int) else resolution[0]
            self.forcing, self.solution, self.grid = generate_poisson_1d(
                n_samples=n_samples, nx=nx, num_modes=num_modes, seed=seed
            )
        elif dim == 2:
            if isinstance(resolution, int):
                nx = ny = resolution
            else:
                ny, nx = resolution
            self.forcing, self.solution, self.x_grid, self.y_grid = generate_poisson_2d(
                n_samples=n_samples, nx=nx, ny=ny, num_modes=num_modes, seed=seed
            )
        else:
            raise ValueError(f"Only dim=1 or dim=2 is supported, got: {dim}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.forcing[idx], self.solution[idx]

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (inputs, targets) tuple for model.fit."""
        return self.forcing, self.solution
