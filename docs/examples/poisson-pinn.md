# 1D Poisson Equation with PINN

This example demonstrates solving a 1D boundary value problem using a Physics-Informed Neural Network with Fourier feature coordinate embedding:
$$-\frac{d^2 u}{dx^2} = \pi^2 \sin(\pi x), \quad x \in [0, 1]$$
with homogeneous Dirichlet boundary conditions:
$$u(0) = 0, \quad u(1) = 0$$

Exact analytical solution: $u^*(x) = \sin(\pi x)$.

Source code: [`examples/pinn_poisson.py`](https://github.com/anas-raza/keras-sciml/blob/main/examples/pinn_poisson.py)

---

## Python Implementation

```python
import keras
import numpy as np
from keras_sciml.models import PINN
from keras_sciml.layers.operators import FourierFeatures

# Collocation coordinates
x_interior = np.linspace(0.01, 0.99, 64)[:, None].astype(np.float32)
x_bc = np.array([[0.0], [1.0]], dtype=np.float32)
x_all = np.concatenate([x_interior, x_bc], axis=0)

# Exact solution targets
y_interior = np.sin(np.pi * x_interior)
y_bc = np.zeros_like(x_bc)
y_all = np.concatenate([y_interior, y_bc], axis=0)

# Build PINN with coordinate embedding
pinn = PINN(
    hidden_layers=3,
    hidden_units=64,
    out_dim=1,
    activation="tanh",
    fourier_features=FourierFeatures(output_dim=32, scale=2.0)
)

pinn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss="mse"
)

pinn.fit(x_all, y_all, epochs=40, batch_size=32)
```

