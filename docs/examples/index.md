# Runnable Examples

The `examples/` directory contains standalone, production-ready scientific machine learning scripts that can be executed directly from the terminal or imported as Python modules.

---

## Benchmark Index

| Script | Physical System | SciML Architecture | Loss / Objective |
| :--- | :--- | :--- | :--- |
| [`fno_darcy_flow.py`](darcy-flow.md) | 2D Darcy Flow through Porous Media | **`FNO2D`** | Relative $L^2$ Loss |
| [`fno_burgers.py`](burgers.md) | 1D Viscous Burgers' Equation | **`FNO1D`** | Relative $L^2$ Loss |
| [`pino_burgers.py`](burgers.md) | 1D Burgers' with PDE Regularization | **`PINO`** (FNO + Residual) | Joint Data + Navier-Stokes Residual |
| [`deeponet_antiderivative.py`](antiderivative.md) | Antiderivative Integral Operator | **`DeepONet`** (Cartesian) | Mean Squared Error |
| [`pinn_poisson.py`](poisson-pinn.md) | 1D Poisson Boundary Value Problem | **`PINN`** + Fourier Features | Interior Collocation + Boundary Loss |
| [`fno_sea_surface_temperature.py`](sea-surface-temperature.md) | Real NOAA Sea Surface Temperature | **`FNO2D`** | Spatiotemporal Forecasting Relative $L^2$ |

---

## Running Examples

All examples are executable from the command line:

```bash
# Run 2D Darcy Flow example with PyTorch
KERAS_BACKEND=torch python examples/fno_darcy_flow.py

# Run 1D Burgers' FNO example with TensorFlow
KERAS_BACKEND=tensorflow python examples/fno_burgers.py

# Run Sea Surface Temperature forecasting with JAX
KERAS_BACKEND=jax python examples/fno_sea_surface_temperature.py
```

Each script also exposes a programmatic helper function:
```python
from examples.fno_darcy_flow import run_experiment

model, history, test_loss = run_experiment(
    n_samples=64,
    resolution=16,
    epochs=20,
    verbose=1
)
```

