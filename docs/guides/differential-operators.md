# Differential Operators

Computing derivatives of spatial fields is essential for expressing PDE residuals, boundary conditions, and conservation laws.

`keras-sciml` provides a suite of differentiable layers in `keras_sciml.layers.differential` supporting both:
1. **Discrete Finite-Difference Discretizations** (for structured grids).
2. **Automatic Differentiation (Autograd)** (for mesh-free coordinate coordinates).

---

## 1. Available Differential Layers

| Layer | Mathematical Operator | Discrete Stencil / Operation |
| :--- | :--- | :--- |
| [`Gradient`](../api/differential.md) | $\nabla u = \left(\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}, \dots\right)$ | Central finite differences with boundary extrapolation |
| [`Laplacian`](../api/differential.md) | $\Delta u = \nabla^2 u = \sum_i \frac{\partial^2 u}{\partial x_i^2}$ | Discrete 3-point (1D) or 5-point (2D) Laplacian stencil |
| [`Divergence`](../api/differential.md) | $\nabla \cdot \mathbf{v} = \sum_i \frac{\partial v_i}{\partial x_i}$ | Divergence of vector velocity/flux fields |
| [`Curl`](../api/differential.md) | $\nabla \times \mathbf{v}$ | 2D scalar curl $\frac{\partial v_y}{\partial x} - \frac{\partial v_x}{\partial y}$ or 3D vector curl |

---

## 2. Using Discrete Differential Layers

Discrete layers compute derivatives directly over structured spatial grids:

```python
import keras
import numpy as np
from keras_sciml.layers.differential import Gradient, Laplacian, Divergence, Curl

# Field u(x, y) on a 32x32 grid with dx = dy = 1.0/31.0
h = 1.0 / 31.0
u = keras.random.normal((4, 32, 32, 1))

# 1. Gradient: returns (batch, 32, 32, 2) where channels are (du/dx, du/dy)
grad_layer = Gradient(dx=h, accuracy=2)
grad_u = grad_layer(u)

# 2. Laplacian: returns (batch, 32, 32, 1) computing d^2u/dx^2 + d^2u/dy^2
laplace_layer = Laplacian(dx=h)
laplace_u = laplace_layer(u)

# 3. Divergence of a 2D vector field v = (v_x, v_y)
v = keras.random.normal((4, 32, 32, 2))
div_layer = Divergence(dx=h)
div_v = div_layer(v)  # shape: (4, 32, 32, 1)

# 4. Curl of a 2D vector field v: returns scalar vorticity
curl_layer = Curl(dx=h)
vorticity = curl_layer(v)  # shape: (4, 32, 32, 1)
```

---

## 3. High-Order Stencils & Accuracy

The `accuracy` parameter allows configuring the order of the finite difference stencil:
* `accuracy=2` (default): Standard 2nd-order central differences:
  $$\frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_{i-1}}{2h} + \mathcal{O}(h^2)$$
* `accuracy=4`: 4th-order 5-point central differences:
  $$\frac{\partial u}{\partial x} \approx \frac{-u_{i+2} + 8u_{i+1} - 8u_{i-1} + u_{i-2}}{12h} + \mathcal{O}(h^4)$$

All boundary points are handled via one-sided forward/backward finite differences matching the interior stencil accuracy order, ensuring no loss of shape or grid shrinkage.
