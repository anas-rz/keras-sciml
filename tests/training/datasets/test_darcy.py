import keras
from keras import layers, ops
import numpy as np
import pytest

from keras_sciml.datasets import DarcyFlowDataset
from keras_sciml.losses import RelativeL2Loss


def test_darcy_dataset_training():
    keras.utils.set_random_seed(42)

    # 2D Darcy flow operator learning: permeability a -> pressure field u
    res = 16
    ds = DarcyFlowDataset(n_samples=24, resolution=res, field_type="piecewise", seed=42)
    a_train, u_train = ds.as_tuple()

    model = keras.Sequential([
        layers.Input(shape=(res, res, 1)),
        layers.Flatten(),
        layers.Dense(64, activation="gelu"),
        layers.Dense(64, activation="gelu"),
        layers.Dense(res * res),
        layers.Reshape((res, res, 1)),
    ])

    loss_fn = RelativeL2Loss()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn)

    initial_loss = float(model.evaluate(a_train, u_train, verbose=0))
    history = model.fit(a_train, u_train, epochs=80, batch_size=12, verbose=0)
    final_loss = history.history["loss"][-1]

    assert final_loss < initial_loss * 0.4

