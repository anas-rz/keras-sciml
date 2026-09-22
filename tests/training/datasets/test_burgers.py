import keras
from keras import layers, ops
import numpy as np
import pytest

from keras_sciml.datasets import BurgersDataset
from keras_sciml.losses import RelativeL2Loss


def test_burgers_dataset_training():
    keras.utils.set_random_seed(42)

    # Burgers operator learning: u_0 -> u_t
    nx = 32
    ds = BurgersDataset(mode="pseudospectral", n_samples=32, nx=nx, t_end=0.2, nu=0.05, seed=42)
    u0_train, ut_train = ds.as_tuple()

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

    initial_loss = float(model.evaluate(u0_train, ut_train, verbose=0))
    history = model.fit(u0_train, ut_train, epochs=80, batch_size=16, verbose=0)
    final_loss = history.history["loss"][-1]

    assert final_loss < initial_loss * 0.4

