# keras-sciml

**Scientific Machine Learning in Keras 3.**

`keras-sciml` is a multi-backend scientific machine learning framework built on Keras 3. It provides reusable scientific layers and modern neural architectures for learning physical systems, differential equations, and operators.

The project focuses on **neural operators, physics-informed learning, and differentiable scientific computing**, while maintaining Keras 3's TensorFlow, PyTorch, and JAX backend compatibility.

## ✨ Features

* 🧮 Differentiable scientific operators
* 🌊 Neural operators for PDE and operator learning
* ⚙️ Physics-informed neural networks
* 🔬 Physics-informed neural operators
* 🔄 Keras 3 multi-backend support
* 🧩 Composable Keras layers and models
* 🧪 Original implementation parity tests
* 🚦 Generic API and shape smoke tests
* 📈 Training tests verifying models actually learn

## 🏗️ Project Structure

```text
keras_sciml/
│
├── layers/
│   ├── operators/
│   │   ├── spectral_conv.py
│   │   ├── fourier_features.py
│   │   └── deeponet.py
│   │
│   ├── differential/
│   │   ├── gradient.py
│   │   ├── laplacian.py
│   │   ├── divergence.py
│   │   └── curl.py
│   │
│   └── physics/
│       ├── boundary_condition.py
│       ├── constraint.py
│       └── residual.py
│
├── models/
│   ├── fno.py
│   ├── deeponet.py
│   ├── pinn.py
│   └── pino.py
│
├── losses/
│   ├── physics_loss.py
│   ├── boundary_loss.py
│   └── conservation_loss.py
│
├── datasets/
│   ├── burgers.py
│   ├── poisson.py
│   └── darcy.py
│
├── metrics/
│   └── relative_l2.py
│
└── tests/
    ├── parity/
    │   ├── layers/
    │   └── models/
    │
    ├── smoke/
    │   ├── layers/
    │   └── models/
    │
    └── training/
        ├── models/
        └── examples/
```

# 🧪 Testing

Testing is a core part of `keras-sciml`.

Every implementation should be validated at three levels.

## 1. Original Implementation Parity

Where an original implementation exists, `keras-sciml` is tested against it.

These tests verify that the Keras implementation produces equivalent results to the reference implementation.

```text
Reference Implementation
        │
        ├── same inputs
        │
        ▼
 ┌───────────────┐
 │ Original      │
 │ Implementation│
 └───────┬───────┘
         │
         │ compare
         ▼
 ┌───────────────┐
 │ keras-sciml   │
 │ Implementation│
 └───────────────┘
```

Parity tests should check, where applicable:

* Output values
* Output shapes
* Parameters / weights
* Numerical tolerance
* Forward-pass behavior
* Important edge cases

For example:

```python
def test_fno_parity():
    reference_output = reference_model(x)
    keras_output = keras_model(x)

    np.testing.assert_allclose(
        keras_output,
        reference_output,
        rtol=1e-5,
        atol=1e-6,
    )
```

The exact tolerance should depend on the backend and reference implementation.

---

## 2. Generic Smoke Tests

Smoke tests verify that every layer and model behaves correctly as a Keras component.

These tests do **not** require a reference implementation.

They should verify:

* The object can be instantiated
* The forward pass works
* Input/output shapes are correct
* Different valid input sizes work where supported
* The layer/model can be serialized
* `get_config()` works where applicable
* Gradients can be computed
* The model can be compiled

Example:

```python
def test_fno_smoke():
    model = FNO(
        modes=8,
        width=32,
        depth=2,
    )

    x = keras.random.normal((2, 64, 64, 3))
    y = model(x)

    assert y.shape == (2, 64, 64, 3)
```

---

## 3. Training Tests

A model being able to execute a forward pass is not enough.

Trainable models should have a small synthetic problem demonstrating that they can actually learn.

Training tests should:

* Use a deterministic synthetic dataset
* Use a small model
* Run quickly on CPU
* Train for only a few iterations/epochs
* Verify that the loss decreases
* Verify that the model reaches a minimum learning threshold

Example:

```python
def test_fno_learns():
    model = FNO(
        modes=8,
        width=16,
        depth=2,
    )

    initial_loss = evaluate_loss(model)

    train(model)

    final_loss = evaluate_loss(model)

    assert final_loss < initial_loss
```

For scientific models, these tests can use simple problems such as:

```text
Burgers Equation
Poisson Equation
Darcy Flow
Diffusion
Advection
```

The goal is **not** to benchmark model performance.

The goal is to verify:

> The implementation can propagate gradients and learn the intended scientific mapping.

---

# 🔬 Testing Matrix

Every trainable component should ideally have:

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

A training test is not necessary for purely deterministic utility/differential layers that have no trainable parameters.

# 🌐 Multi-Backend Testing

The same test suite should be executed across Keras 3 backends:

```text
                 Test Suite
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     TensorFlow   PyTorch      JAX
```

The objective is to ensure that the public `keras-sciml` API behaves consistently across backends.

## 🎯 Philosophy

`keras-sciml` is not intended to be a collection of unrelated scientific models.

The project is built around four principles:

**Composable**

Scientific operations should be usable as ordinary Keras layers.

**Multi-backend**

Models should leverage Keras 3 rather than being tied to a single ML framework.

**Physics-aware**

Physical laws, differential equations, constraints, and conservation principles should be first-class components.

**Verified**

Implementations should demonstrate both **numerical correctness** and **actual learning behavior**.

## 🗺️ Roadmap

### v0.1

* [ ] Fourier Neural Operator
* [ ] DeepONet
* [ ] PINN
* [ ] PINO
* [ ] Spectral convolution
* [ ] Fourier features
* [ ] Gradient
* [ ] Laplacian
* [ ] Divergence
* [ ] Curl
* [ ] Physics and boundary losses
* [ ] Canonical PDE datasets
* [ ] Parity tests
* [ ] Smoke tests
* [ ] Training tests
* [ ] TensorFlow backend

### v0.2

* [ ] AFNO
* [ ] UNO
* [ ] Neural Operator Transformer
* [ ] Expanded PDE operators
* [ ] PyTorch backend
* [ ] JAX backend

### Future

* Scientific graph neural networks
* Equivariant scientific models
* Molecular ML
* Materials ML
* Differentiable numerical solvers

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.

---

**keras-sciml**
*Scientific Machine Learning, built with Keras 3.*
