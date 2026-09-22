import keras
import numpy as np
import pytest

from keras_sciml.datasets import PoissonDataset
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import FNO1D


def test_fno1d_learns():
    keras.utils.set_random_seed(42)

    # 1D Poisson operator dataset: forcing f -> solution u
    nx = 32
    n_samples = 32
    ds = PoissonDataset(dim=1, n_samples=n_samples, resolution=nx, num_modes=4, seed=42)
    f_train, u_train = ds.as_tuple()

    model = FNO1D(modes=8, width=16, depth=2, out_channels=1)
    loss_fn = RelativeL2Loss()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss=loss_fn)

    initial_loss = float(model.evaluate(f_train, u_train, verbose=0))
    history = model.fit(f_train, u_train, epochs=60, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]

    # Verify significant learning
    assert final_loss < initial_loss * 0.3

