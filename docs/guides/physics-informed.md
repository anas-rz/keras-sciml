# Physics-Informed Learning

Physics-informed machine learning integrates mathematical models (governing PDEs, boundary conditions, and conservation principles) directly into the neural network architecture and training objectives.

`keras-sciml` supports two complementary physics-informed approaches:
1. **Physics-Informed Neural Networks (`PINN`)**: Coordinate-based networks parameterized as continuous functions over space-time: $u_\theta(\mathbf{x}, t)$.
2. **Physics-Informed Neural Operators (`PINO`)**: Operator networks that learn mapping between function spaces while regularizing predicted solution fields using physical PDE residuals.

---

## 1. Physics-Informed Neural Networks (PINN)

A standard PINN maps spatio-temporal coordinates $\mathbf{x} \in \mathbb{R}^d$ to solution values $u(\mathbf{x})$.

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{data}} + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}}$$

```text
Coordinates (x, y, t) ───► FourierFeatures ───► MLP ───► u(x, y, t)
                                                           │
                                                           ├───► PDE Residual R(u)
                                                           └───► Boundary Conditions
```

### Mitigating Spectral Bias with Fourier Features

Standard MLPs suffer from *spectral bias*—a tendency to learn low frequencies much faster than high frequencies. To capture sharp wavefronts and high-frequency harmonics, pass a [`FourierFeatures`](../api/operators.md) layer to `PINN`:

$$\gamma(\mathbf{x}) = \left[\cos(2\pi B \mathbf{x}), \sin(2\pi B \mathbf{x})\right]$$

```python
from keras_sciml.models import PINN
from keras_sciml.layers.operators import FourierFeatures

# PINN with Gaussian random Fourier feature projection
model = PINN(
    hidden_layers=4,
    hidden_units=64,
    out_dim=1,
    activation="tanh",
    fourier_features=FourierFeatures(output_dim=32, scale=2.0)
)
```

---

## 2. Physics-Informed Neural Operators (PINO)

While PINNs train a single network to solve one specific PDE instance, **Physics-Informed Neural Operators (PINO)** learn an entire family of solutions across varying coefficients or initial conditions, regularized by physical residuals:

$$\mathcal{L} = \mathcal{L}_{\text{data}}(G_\theta(a), u) + \lambda_{\text{pde}} \|\mathcal{R}(G_\theta(a))\|^2 + \lambda_{\text{bc}} \|\mathcal{B}(G_\theta(a))\|^2$$

```python
from keras_sciml.models import PINO, FNO1D
from keras_sciml.layers.differential import Gradient, Laplacian
from keras_sciml.losses import RelativeL2Loss

# 1. Define physical PDE residual for viscous Burgers' equation:
#    R(u) = u * du/dx - nu * d^2u/dx^2
grad = Gradient(dx=1.0/63.0)
laplace = Laplacian(dx=1.0/63.0)
nu = 0.01

def burgers_residual(u_pred, a_input):
    u_x = grad(u_pred)
    u_xx = laplace(u_pred)
    residual = u_pred * u_x - nu * u_xx
    return residual

# 2. Wrap FNO backbone in PINO
model = PINO(
    backbone=FNO1D(modes=16, width=64, depth=4),
    pde_fn=burgers_residual,
    pde_weight=0.1
)

model.compile(
    optimizer="adam",
    loss=RelativeL2Loss()
)
```

---

## 3. Boundary & Conservation Constraints

* [`BoundaryCondition`](../api/physics.md): Enforces Dirichlet, Neumann, or Robin boundary conditions on predictions.
* [`BoundaryLoss`](../api/losses.md): Explicit penalty loss measuring deviation on domain boundaries.
* [`ConservationLoss`](../api/losses.md): Enforces physical invariants (mass, momentum, or energy conservation $\int_\Omega u \, dx = C$).
