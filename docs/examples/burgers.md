# 1D Burgers' Equation (FNO & PINO)

The 1D viscous Burgers' equation is a fundamental non-linear PDE modeling fluid turbulence, shock formation, and wave propagation:
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

`keras-sciml` provides two example approaches for learning this system:
1. Pure data-driven operator learning via **`FNO1D`** ([`examples/fno_burgers.py`](https://github.com/anas-raza/keras-sciml/blob/main/examples/fno_burgers.py)).
2. Physics-Informed Neural Operator via **`PINO`** ([`examples/pino_burgers.py`](https://github.com/anas-raza/keras-sciml/blob/main/examples/pino_burgers.py)).

---

## 1. Pure Data-Driven: FNO1D

Maps initial velocity profile $u(x, t=0)$ to future solution $u(x, t=1)$ on a periodic 1D grid.

```python
import keras
from keras_sciml.datasets import BurgersDataset
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import FNO1D

dataset = BurgersDataset(n_samples=50, nx=64, viscosity=0.01)
u0, u1 = dataset.as_tuple()

model = FNO1D(
    modes=16,
    width=64,
    depth=4,
    out_channels=1,
    activation="gelu"
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss=RelativeL2Loss(),
)

model.fit(u0, u1, epochs=25, batch_size=16)
```

---

## 2. Physics-Informed: PINO

Combines the `FNO1D` backbone with a PDE residual loss computed using discrete `Gradient` and `Laplacian` layers:
$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{pde}} \left\| u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} \right\|^2$$

```python
import keras
from keras_sciml.datasets import BurgersDataset
from keras_sciml.layers.differential import Gradient, Laplacian
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import FNO1D, PINO

dataset = BurgersDataset(n_samples=32, nx=32, viscosity=0.01)
u0, u1 = dataset.as_tuple()

grad = Gradient(dx=1.0 / 31.0)
laplace = Laplacian(dx=1.0 / 31.0)

def burgers_pde_residual(u_pred, a_input):
    return u_pred * grad(u_pred) - 0.01 * laplace(u_pred)

pino = PINO(
    backbone=FNO1D(modes=8, width=32, depth=3),
    pde_fn=burgers_pde_residual,
    pde_weight=0.05
)

pino.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss=RelativeL2Loss()
)

pino.fit(u0, u1, epochs=20, batch_size=8)
```

