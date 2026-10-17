# Scientific Datasets

`keras-sciml` bundles fast numerical solvers for canonical benchmark PDEs, as well as real-world observational data loaders.

All datasets provide:
1. Fast functional data generation (`generate_*`).
2. High-level dataset container classes (`*Dataset`) with `.as_tuple()`, indexing, and len methods.

---

## 1. 2D Darcy Flow (`darcy.py`)

Solves the steady-state diffusion equation with variable permeability coefficient field $a(x, y)$:
$$-\nabla \cdot (a(x) \nabla u(x)) = f(x), \quad x \in [0, 1]^2, \quad u|_{\partial \Omega} = 0$$

* Input: Permeability field $a(x, y)$ generated from Gaussian Random Fields (piecewise or log-normal).
* Target: Piezometric head / pressure solution $u(x, y)$ solved via 5-point finite difference with Conjugate Gradient.

```python
from keras_sciml.datasets import DarcyFlowDataset, generate_darcy_2d

dataset = DarcyFlowDataset(
    n_samples=100,
    resolution=32,
    field_type="piecewise", # or "lognormal"
    seed=42
)
a, u = dataset.as_tuple()
print("Permeability shape:", a.shape)  # (100, 32, 32, 1)
print("Solution shape:    ", u.shape)  # (100, 32, 32, 1)
```

---

## 2. 1D Viscous Burgers' Equation (`burgers.py`)

Solves the non-linear 1D advection-diffusion equation:
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, 1), \quad t \in [0, 1]$$

* Input: Initial velocity profile $u(x, t=0)$ sampled from periodic Gaussian Random Fields.
* Target: Future velocity profile $u(x, t=1)$ solved using pseudo-spectral spatial differentiation and Runge-Kutta 4th order (RK4) time integration.

```python
from keras_sciml.datasets import BurgersDataset

dataset = BurgersDataset(n_samples=200, nx=64, viscosity=0.01)
u0, u1 = dataset.as_tuple()
```

---

## 3. 1D/2D Poisson Equation (`poisson.py`)

Solves the elliptic boundary value problem:
$$-\Delta u = f(x), \quad x \in \Omega, \quad u|_{\partial \Omega} = 0$$

```python
from keras_sciml.datasets import PoissonDataset

# 2D Poisson with exact analytical source forcing
dataset_2d = PoissonDataset(n_samples=50, resolution=32, dim=2)
f, u = dataset_2d.as_tuple()
```

---

## 4. Real Observational Data: NOAA Sea Surface Temperature (`sea_surface_temperature.py`)

Real-world observational measurements collected from NOAA satellite radiometers (AVHRR) and in-situ ocean buoys in the North Atlantic.

* Shape: `(50, 64, 64, 6)` representing 50 sequential trajectories across 6 consecutive time steps.
* Features:
  - Automatic download and caching via `keras.utils.get_file`.
  - Offline physical synthetic fallback if internet is not available.
  - Slicing into sliding window temporal pairs for autoregressive forecasting.

```python
from keras_sciml.datasets import SeaSurfaceTemperatureDataset, load_sea_surface_temperature

x_train, y_train, x_test, y_test, meta = load_sea_surface_temperature(
    download=True,
    time_history=1,
    time_future=1,
    step_mode="pairs",
    normalize=True
)
print("Data source:", meta["source"])
print("Train shape:", x_train.shape)
```

