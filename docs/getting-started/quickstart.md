# Quickstart

This 5-minute guide demonstrates two fundamental paradigms in scientific machine learning using `keras-sciml`:
1. **Neural Operator Learning (FNO)**: Learning a continuous mapping between function spaces.
2. **Physics-Informed Learning (PINN)**: Solving a boundary value problem governed by a differential equation.

---

## 1. Neural Operator: 1D Fourier Neural Operator

Neural operators learn mappings between infinite-dimensional function spaces. Here we train an `FNO1D` on the canonical 1D viscous Burgers' equation: mapping initial velocity profiles $u(x, t=0)$ to future states $u(x, t=1)$.

```python
import keras
from keras_sciml.datasets import BurgersDataset
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import FNO1D

# 1. Load 1D Burgers' equation dataset
dataset = BurgersDataset(n_samples=200, nx=64, viscosity=0.01)
u0_train, u1_train = dataset.as_tuple()

# 2. Build 1D Fourier Neural Operator
model = FNO1D(
    modes=16,          # Retain lowest 16 Fourier frequencies
    width=64,          # Latent channel width
    depth=4,           # 4 Fourier spectral convolution layers
    out_channels=1,    # Output channels
    activation="gelu"
)

# 3. Compile with relative L2 loss (standard for operator learning)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=RelativeL2Loss(),
    metrics=[RelativeL2Loss(name="rel_l2")]
)

# 4. Train the operator
model.fit(u0_train, u1_train, epochs=15, batch_size=16)

# 5. Evaluate on new initial condition
preds = model.predict(u0_train[:5])
print("Predictions shape:", preds.shape)
```

---

## 2. Physics-Informed Neural Network (PINN)

PINNs embed physical equations directly into the loss function using automatic differentiation or differential layers. Here we solve the 1D Poisson boundary value problem:
$$-\frac{d^2 u}{dx^2} = \pi^2 \sin(\pi x), \quad x \in [0, 1], \quad u(0) = u(1) = 0$$

The exact analytical solution is $u(x) = \sin(\pi x)$.

```python
import keras
import numpy as np
from keras_sciml.models import PINN
from keras_sciml.layers.operators import FourierFeatures

# 1. Generate collocation points
x_collocation = np.linspace(0.0, 1.0, 64)[:, None].astype(np.float32)
y_analytical = np.sin(np.pi * x_collocation)

# 2. Build PINN with coordinate embedding
model = PINN(
    hidden_layers=3,
    hidden_units=64,
    out_dim=1,
    activation="tanh",
    fourier_features=FourierFeatures(output_dim=32, scale=2.0)
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)

# 3. Train on collocation points
model.fit(x_collocation, y_analytical, epochs=30, batch_size=32)

# 4. Test predictions
u_pred = model.predict(x_collocation)
l2_err = np.linalg.norm(u_pred - y_analytical) / np.linalg.norm(y_analytical)
print(f"Relative L2 error vs exact solution: {l2_err:.4e}")
```

---

## 3. Next Steps

* Explore [Neural Operators](../guides/neural-operators.md) for 2D Darcy Flow and wave equations.
* Explore [Differential Operators](../guides/differential-operators.md) to build custom PDE residual layers.
* Check out the [Runnable Examples](../examples/index.md) in the repository.
