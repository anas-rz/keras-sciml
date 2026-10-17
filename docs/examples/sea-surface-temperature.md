# Sea Surface Temperature (NOAA OISST) Forecasting

This example demonstrates training an `FNO2D` model on real-world satellite radiometer and in-situ buoy observations of Sea Surface Temperature in the North Atlantic.

Source code: [`examples/fno_sea_surface_temperature.py`](https://github.com/anas-raza/keras-sciml/blob/main/examples/fno_sea_surface_temperature.py)

---

## The Physical Problem

Global ocean temperatures are governed by turbulent thermal advection, Coriolis drift, and surface heat fluxes:
$$\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \kappa \nabla^2 T + Q$$

The goal is to learn the spatiotemporal operator mapping the 2D surface temperature field at time $t$ to future temperature at time $t + \Delta t$:
$$G: T(\cdot, t) \mapsto T(\cdot, t + \Delta t)$$

---

## Python Implementation

```python
import keras
from keras_sciml.datasets import load_sea_surface_temperature
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import FNO2D

# 1. Fetch real NOAA SST data (or synthetic physical fallback if offline)
x_train, y_train, x_test, y_test, meta = load_sea_surface_temperature(
    download=True,
    time_history=1,
    time_future=1,
    step_mode="pairs",
    normalize=True
)

# 2. Build 2D Fourier Neural Operator
model = FNO2D(
    modes=(8, 8),
    width=24,
    depth=3,
    out_channels=1,
    activation="gelu"
)

# 3. Train model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss=RelativeL2Loss(),
    metrics=[RelativeL2Loss(name="rel_l2"), "mae"]
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=25,
    batch_size=16
)
```

