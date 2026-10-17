# 2D Darcy Flow with FNO2D

This example demonstrates training a 2D Fourier Neural Operator to learn the non-linear mapping from permeability coefficient field $a(x, y)$ to steady-state fluid pressure $u(x, y)$ in porous media:
$$-\nabla \cdot (a(x) \nabla u(x)) = f(x), \quad x \in [0, 1]^2, \quad u|_{\partial \Omega} = 0$$

Source code: [`examples/fno_darcy_flow.py`](https://github.com/anas-raza/keras-sciml/blob/main/examples/fno_darcy_flow.py)

---

## Architecture

* **Lifting Layer**: `Dense(width=32)` lifting scalar permeability to 32 latent channels.
* **Spectral Layers**: 4 blocks of `SpectralConv2D(modes=(12, 12), channels=32)` with residual skip connections and `gelu` non-linearities.
* **Projection Layer**: 2-layer MLP projecting 32 latent channels to 1 output channel.
* **Loss Function**: `RelativeL2Loss()` measuring $\|u_{\text{pred}} - u_{\text{true}}\|_2 / \|u_{\text{true}}\|_2$.

---

## Python Implementation

```python
import keras
from keras_sciml.datasets import DarcyFlowDataset
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import FNO2D

# Load dataset
dataset = DarcyFlowDataset(n_samples=64, resolution=16, field_type="piecewise", seed=42)
a_data, u_data = dataset.as_tuple()

# Train / test split (80/20)
n_train = int(0.8 * len(dataset))
a_train, u_train = a_data[:n_train], u_data[:n_train]
a_test, u_test = a_data[n_train:], u_data[n_train:]

# Build FNO2D
model = FNO2D(
    modes=(8, 8),
    width=32,
    depth=3,
    out_channels=1,
    activation="gelu"
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=RelativeL2Loss(),
    metrics=[RelativeL2Loss(name="rel_l2")]
)

history = model.fit(
    a_train,
    u_train,
    validation_data=(a_test, u_test),
    epochs=30,
    batch_size=16
)
```

