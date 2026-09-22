import keras
from keras import layers, ops
import numpy as np
import pytest

from keras_sciml.datasets import PoissonDataset
from keras_sciml.losses import RelativeL2Loss


def test_poisson_dataset_training():
    keras.utils.set_random_seed(42)

    # 1D Poisson operator dataset: forcing f -> solution u
    nx = 32
    ds = PoissonDataset(dim=1, n_samples=32, resolution=nx, num_modes=4, seed=42)
    f_train, u_train = ds.as_tuple()

    model = keras.Sequential([
        layers.Input(shape=(nx, 1)),
        layers.Flatten(),
        layers.Dense(64, activation="gelu"),
        layers.Dense(64, activation="gelu"),
        layers.Dense(nx),
        layers.Reshape((nx, 1)),
    ])

    loss_fn = RelativeL2Loss()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)

    initial_loss = float(model.evaluate(f_train, u_train, verbose=0))
    history = model.fit(f_train, u_train, epochs=80, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]

    assert final_loss < initial_loss * 0.3

