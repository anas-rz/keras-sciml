# keras-sciml

[![PyPI](https://img.shields.io/pypi/v/keras-sciml)](https://pypi.org/project/keras-sciml/)
[![Python](https://img.shields.io/pypi/pyversions/keras-sciml)](https://pypi.org/project/keras-sciml/)
[![Tests](https://github.com/anas-rz/keras-sciml/actions/workflows/tests.yml/badge.svg)](https://github.com/anas-rz/keras-sciml/actions/workflows/tests.yml)
[![Docs](https://github.com/anas-rz/keras-sciml/actions/workflows/docs.yml/badge.svg)](https://anas-rz.github.io/keras-sciml/)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

**Scientific Machine Learning in Keras 3.**

`keras-sciml` is a multi-backend scientific machine learning framework built on Keras 3. It provides reusable scientific layers and modern neural architectures for learning physical systems, differential equations, and operators.

The project focuses on **neural operators, physics-informed learning, and differentiable scientific computing**, and runs unchanged on TensorFlow, PyTorch, and JAX.

📖 **Documentation:** [anas-rz.github.io/keras-sciml](https://anas-rz.github.io/keras-sciml/)

## ✨ Features

* 🌊 **Neural operators**: Fourier Neural Operators (`FNO1D`, `FNO2D`) and `DeepONet`
* ⚙️ **Physics-informed models**: `PINN` and `PINO` trained on PDE residuals
* 🧮 **Differentiable operators**: `Gradient`, `Laplacian`, `Divergence`, `Curl` as Keras layers
* 🧱 **Physics layers**: boundary conditions, constraints, and PDE residuals (Poisson, diffusion, Burgers, advection)
* 📉 **Scientific losses**: relative L2, H1 Sobolev, physics, boundary, and conservation losses
* 🗂️ **Datasets**: Burgers, Darcy flow, Poisson, and NOAA sea surface temperature
* 🔄 **Keras 3 multi-backend**: TensorFlow, PyTorch, and JAX
* 🧪 **Verified**: parity, smoke, and training tests on every backend

## 📦 Installation

```bash
pip install keras-sciml
```

`keras-sciml` needs Python 3.10+ and a Keras 3 backend. Install at least one:

```bash
pip install "jax[cpu]"        # JAX
pip install torch             # PyTorch
pip install tensorflow        # TensorFlow
```

Then select it with the `KERAS_BACKEND` environment variable before importing Keras:

```bash
export KERAS_BACKEND="jax"    # or "torch", "tensorflow"
```

To install from source for development:

```bash
git clone https://github.com/anas-rz/keras-sciml.git
cd keras-sciml
pip install -e ".[test]"      # add ",docs" to build the documentation
```

## 🚀 Quickstart

Train a Fourier Neural Operator that maps an initial condition of the viscous Burgers' equation to its solution at a later time:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow", "torch"

import keras
from keras_sciml.datasets import BurgersDataset
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import FNO1D

# Initial conditions u(x, 0) and solutions u(x, T) of the viscous Burgers' equation
u0, ut = BurgersDataset(n_samples=64, nx=64, nu=0.02).as_tuple()

model = FNO1D(modes=12, width=32, depth=3, out_channels=1)
model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=RelativeL2Loss())
model.fit(u0, ut, epochs=20, batch_size=16)

print(model.predict(u0[:4]).shape)  # (4, 64, 1)
```

## 📓 Examples

Every example is a notebook you can open in Colab. Browse them all in the [examples gallery](https://anas-rz.github.io/keras-sciml/examples/).

| Example | Model | Notebook |
| --- | --- | --- |
| 2D Darcy flow | `FNO2D` | [examples/neural_operators/fno_darcy_flow.ipynb](https://github.com/anas-rz/keras-sciml/blob/main/examples/neural_operators/fno_darcy_flow.ipynb) |
| 1D Burgers' equation | `FNO1D` | [examples/neural_operators/fno_burgers.ipynb](https://github.com/anas-rz/keras-sciml/blob/main/examples/neural_operators/fno_burgers.ipynb) |
| Antiderivative operator | `DeepONet` | [examples/neural_operators/deeponet_antiderivative.ipynb](https://github.com/anas-rz/keras-sciml/blob/main/examples/neural_operators/deeponet_antiderivative.ipynb) |
| 1D Poisson equation | `PINN` | [examples/physics_informed/pinn_poisson.ipynb](https://github.com/anas-rz/keras-sciml/blob/main/examples/physics_informed/pinn_poisson.ipynb) |
| 1D Burgers' equation | `PINO` | [examples/physics_informed/pino_burgers.ipynb](https://github.com/anas-rz/keras-sciml/blob/main/examples/physics_informed/pino_burgers.ipynb) |
| Sea surface temperature forecasting | `FNO2D` | [examples/climate/fno_sea_surface_temperature.ipynb](https://github.com/anas-rz/keras-sciml/blob/main/examples/climate/fno_sea_surface_temperature.ipynb) |

## 🏗️ Project Structure

```text
keras_sciml/
├── layers/
│   ├── operators/        # SpectralConv1D/2D, FourierFeatures, DeepONet layers
│   ├── differential/     # Gradient, Laplacian, Divergence, Curl
│   └── physics/          # boundary conditions, constraints, PDE residuals
├── models/               # FNO1D, FNO2D, DeepONet, PINN, PINO
├── losses/               # RelativeL2Loss, H1Loss, PhysicsLoss, BoundaryLoss, ConservationLoss
└── datasets/             # Burgers, Darcy flow, Poisson, sea surface temperature

examples/                 # notebooks, one folder per category
tests/
├── parity/               # match reference implementations
├── smoke/                # Keras API, shapes, serialization, gradients
└── training/             # models learn on small problems; examples run end to end
```

# 🧪 Testing

Testing is a core part of `keras-sciml`. Every implementation is validated at three levels, and the whole suite runs on TensorFlow, PyTorch, and JAX in CI.

```bash
KERAS_BACKEND=jax pytest tests/
```

## 1. Original Implementation Parity

Where an original implementation exists, `keras-sciml` is tested against it with the same inputs. Parity tests check, where applicable:

* Output values and shapes
* Parameters / weights
* Forward-pass behavior within a numerical tolerance
* Important edge cases

```python
def test_fno_parity():
    reference_output = reference_model(x)
    keras_output = keras_model(x)

    np.testing.assert_allclose(keras_output, reference_output, rtol=1e-5, atol=1e-6)
```

The exact tolerance depends on the backend and reference implementation.

## 2. Generic Smoke Tests

Smoke tests verify that every layer and model behaves correctly as a Keras component, without needing a reference implementation:

* The object can be instantiated and run forward
* Input/output shapes are correct, including different valid input sizes
* `get_config()` and serialization work
* Gradients can be computed
* The model can be compiled

## 3. Training Tests

A forward pass is not enough. Trainable models are trained on a small, deterministic, synthetic problem that runs quickly on CPU, and the test checks that the loss decreases. The example notebooks are also run end to end with reduced sizes.

The goal is **not** to benchmark model performance. It is to verify:

> The implementation can propagate gradients and learn the intended scientific mapping.

## 🔬 Testing Matrix

| Component           | Parity | Smoke | Training |
| ------------------- | -----: | ----: | -------: |
| SpectralConv        |      ✓ |     ✓ |        ✓ |
| FourierFeatures     |      ✓ |     ✓ |        ✓ |
| DeepONet            |      ✓ |     ✓ |        ✓ |
| FNO                 |      ✓ |     ✓ |        ✓ |
| PINN                |      ✓ |     ✓ |        ✓ |
| PINO                |      ✓ |     ✓ |        ✓ |
| Gradient            |      ✓ |     ✓ |        — |
| Laplacian           |      ✓ |     ✓ |        — |
| Divergence          |      ✓ |     ✓ |        — |
| Curl                |      ✓ |     ✓ |        — |
| Physics losses      |      ✓ |     ✓ |        ✓ |
| Boundary conditions |      ✓ |     ✓ |        ✓ |

Purely deterministic differential layers have no trainable parameters, so they have no training test.

## 🎯 Philosophy

`keras-sciml` is not intended to be a collection of unrelated scientific models. The project is built around four principles:

* **Composable**: scientific operations are ordinary Keras layers.
* **Multi-backend**: models use Keras 3 rather than being tied to a single ML framework.
* **Physics-aware**: physical laws, differential equations, constraints, and conservation principles are first-class components.
* **Verified**: implementations demonstrate both **numerical correctness** and **actual learning behavior**.

## 🗺️ Roadmap

### v0.1

* [x] Fourier Neural Operator
* [x] DeepONet
* [x] PINN
* [x] PINO
* [x] Spectral convolution
* [x] Fourier features
* [x] Gradient, Laplacian, Divergence, Curl
* [x] Physics and boundary losses
* [x] Canonical PDE datasets
* [x] Parity, smoke, and training tests
* [x] TensorFlow, PyTorch, and JAX backends

### v0.2

* [ ] AFNO
* [ ] UNO
* [ ] Neural Operator Transformer
* [ ] Expanded PDE operators

### Future

* Scientific graph neural networks
* Equivariant scientific models
* Molecular ML
* Materials ML
* Differentiable numerical solvers

## 🤝 Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request. See the [contributing guide](https://anas-rz.github.io/keras-sciml/developer/contributing/).

## 📄 License

MIT License

---

**keras-sciml**
*Scientific Machine Learning, built with Keras 3.*
