from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def sample_gaussian_random_field_2d(
    n_samples: int,
    nx: int,
    ny: int,
    alpha: float = 2.5,
    tau: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate 2D Gaussian random fields using spectral covariance: C = (-Delta + tau^2 I)^(-alpha)."""
    if rng is None:
        rng = np.random.default_rng(42)

    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    # Eigenvalues of Laplacian: 4 * pi^2 * (kx^2 + ky^2)
    laplace_eig = 4.0 * (np.pi ** 2) * (KX ** 2 + KY ** 2)
    power_spectrum = (laplace_eig + (tau ** 2)) ** (-alpha / 2.0)
    # Zero out zero frequency to ensure zero mean
    power_spectrum[0, 0] = 0.0

    # Complex standard Gaussian noise: (n_samples, nx, ny)
    noise_real = rng.standard_normal((n_samples, nx, ny)).astype(np.float32)
    noise_imag = rng.standard_normal((n_samples, nx, ny)).astype(np.float32)
    noise_hat = (noise_real + 1j * noise_imag) * power_spectrum[None, :, :]

    grf = np.fft.ifft2(noise_hat, axes=(1, 2)).real.astype(np.float32)
    # Standardize
    grf = grf / (np.std(grf, axis=(1, 2), keepdims=True) + 1e-6)
    return grf


def solve_darcy_2d_sample(
    a: np.ndarray,
    f_val: float = 1.0,
) -> np.ndarray:
    """Solve 2D Darcy flow -nabla cdot (a nabla u) = f on a single grid with u=0 on boundary.

    Uses 5-point finite difference discretization and conjugate gradient solver.

    Args:
        a: ndarray of shape `(ny, nx)`, permeability field (a > 0).
        f_val: float, constant forcing term f.

    Returns:
        ndarray of shape `(ny, nx)`, solution field u.
    """
    ny, nx = a.shape
    h = 1.0 / (nx - 1)
    h2 = h ** 2

    # Interior indices
    n_interior_x = nx - 2
    n_interior_y = ny - 2
    n_dofs = n_interior_x * n_interior_y

    if n_dofs <= 0:
        return np.zeros_like(a)

    def to_dof(i, j):
        return (i - 1) * n_interior_x + (j - 1)

    # Harmonic mean for interface permeability
    # Arithmetic: 0.5 * (a1 + a2)
    rows = []
    cols = []
    data = []
    rhs = np.full(n_dofs, f_val * h2, dtype=np.float64)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            idx = to_dof(i, j)

            # Interface conductivities
            a_east = 0.5 * (a[i, j] + a[i, j + 1])
            a_west = 0.5 * (a[i, j] + a[i, j - 1])
            a_north = 0.5 * (a[i, j] + a[i + 1, j])
            a_south = 0.5 * (a[i, j] + a[i - 1, j])

            diag = a_east + a_west + a_north + a_south

            rows.append(idx)
            cols.append(idx)
            data.append(diag)

            # East neighbor: (i, j + 1)
            if j + 1 < nx - 1:
                rows.append(idx)
                cols.append(to_dof(i, j + 1))
                data.append(-a_east)

            # West neighbor: (i, j - 1)
            if j - 1 > 0:
                rows.append(idx)
                cols.append(to_dof(i, j - 1))
                data.append(-a_west)

            # North neighbor: (i + 1, j)
            if i + 1 < ny - 1:
                rows.append(idx)
                cols.append(to_dof(i + 1, j))
                data.append(-a_north)

            # South neighbor: (i - 1, j)
            if i - 1 > 0:
                rows.append(idx)
                cols.append(to_dof(i - 1, j))
                data.append(-a_south)

    A_mat = sp.csr_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))

    # Solve symmetric positive-definite linear system with CG
    u_dofs, info = spla.cg(A_mat, rhs, rtol=1e-6, maxiter=500)

    u_full = np.zeros((ny, nx), dtype=np.float32)
    u_full[1:-1, 1:-1] = u_dofs.reshape((n_interior_y, n_interior_x)).astype(np.float32)
    return u_full


def generate_darcy_2d(
    n_samples: int = 50,
    resolution: int = 16,
    field_type: str = "piecewise",
    f_val: float = 1.0,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D Darcy Flow dataset: -nabla cdot (a(x) nabla u(x)) = f(x) with u=0 on boundary.

    Matches the canonical Fourier Neural Operator (FNO) Darcy flow benchmark (Li et al., 2020).

    Args:
        n_samples: int, number of samples to generate.
        resolution: int, spatial grid size (nx = ny = resolution).
        field_type: str, permeability distribution type:
            - `"piecewise"`: Binary permeability field: $a(x) = 12 \\cdot \\mathbb{I}(g>0) + 3 \\cdot \\mathbb{I}(g\\le 0)$
            - `"lognormal"`: Continuous log-normal field: $a(x) = \\exp(g(x))$
        f_val: float, source term constant. Defaults to 1.0.
        seed: Optional[int], random seed.

    Returns:
        Tuple of (permeability, solution, x_grid, y_grid):
        - permeability: ndarray of shape `(n_samples, resolution, resolution, 1)`
        - solution: ndarray of shape `(n_samples, resolution, resolution, 1)`
        - x_grid: ndarray of shape `(resolution,)`
        - y_grid: ndarray of shape `(resolution,)`
    """
    rng = np.random.default_rng(seed)
    nx = ny = resolution
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)

    # Sample underlying Gaussian random fields
    grf = sample_gaussian_random_field_2d(n_samples, nx, ny, rng=rng)

    if field_type == "piecewise":
        # Standard FNO benchmark: a(x) in {3.0, 12.0}
        a_fields = np.where(grf > 0.0, 12.0, 3.0).astype(np.float32)
    elif field_type == "lognormal":
        a_fields = np.exp(grf).astype(np.float32)
    else:
        raise ValueError(f"Unknown field_type: {field_type}. Must be 'piecewise' or 'lognormal'.")

    # Solve for each sample
    u_solutions = np.empty_like(a_fields)
    for i in range(n_samples):
        u_solutions[i] = solve_darcy_2d_sample(a_fields[i], f_val=f_val)

    return a_fields[:, :, :, None], u_solutions[:, :, :, None], x, y


class DarcyFlowDataset:
    """Convenience dataset wrapper for 2D Darcy Flow."""

    def __init__(
        self,
        n_samples: int = 50,
        resolution: int = 16,
        field_type: str = "piecewise",
        f_val: float = 1.0,
        seed: Optional[int] = 42,
    ):
        self.n_samples = n_samples
        self.resolution = resolution
        self.permeability, self.solution, self.x_grid, self.y_grid = generate_darcy_2d(
            n_samples=n_samples,
            resolution=resolution,
            field_type=field_type,
            f_val=f_val,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.permeability[idx], self.solution[idx]

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (inputs, targets) tuple."""
        return self.permeability, self.solution
