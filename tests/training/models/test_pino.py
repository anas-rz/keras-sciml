import keras
from keras import ops
import numpy as np
import pytest

from keras_sciml.datasets import PoissonDataset
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import PINO


def test_pino_learns():
    keras.utils.set_random_seed(42)

    # 1D Poisson operator dataset: f -> u
    nx = 32
    ds = PoissonDataset(dim=1, n_samples=32, resolution=nx, num_modes=4, seed=42)
    f_train, u_train = ds.as_tuple()

    # Simple PDE residual penalty: R(u, f) = u - target
    def simple_pde_res(u, inputs):
        # inputs is forcing f
        return u - inputs

    model = PINO(modes=8, width=16, depth=2, out_channels=1, residual_fn=simple_pde_res, pde_weight=0.01)

    loss_fn = RelativeL2Loss()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss=loss_fn)

    initial_loss = float(model.evaluate(f_train, u_train, verbose=0))
    history = model.fit(f_train, u_train, epochs=60, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]

    # Verify significant learning
    assert final_loss < initial_loss * 0.4

