"""2D Darcy Flow Operator Learning with Fourier Neural Operator (FNO).

This example demonstrates training an FNO2D model to learn the mapping from
permeability field a(x) to pressure solution field u(x):
    -nabla cdot (a(x) nabla u(x)) = f(x),  x in [0, 1]^2,  u|_{partial Omega} = 0
"""

import keras
import numpy as np

from keras_sciml.datasets import DarcyFlowDataset
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import FNO2D


def run_experiment(
    n_samples: int = 64,
    resolution: int = 16,
    modes: int = 8,
    width: int = 32,
    depth: int = 3,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 0.01,
    verbose: int = 1,
):
    keras.utils.set_random_seed(42)

    if verbose:
        print(f"Generating 2D Darcy flow dataset ({n_samples} samples at {resolution}x{resolution})...")
    ds = DarcyFlowDataset(n_samples=n_samples, resolution=resolution, field_type="piecewise", seed=42)
    a_data, u_data = ds.as_tuple()

    # Train / test split (80% / 20%)
    n_train = int(0.8 * n_samples)
    a_train, u_train = a_data[:n_train], u_data[:n_train]
    a_test, u_test = a_data[n_train:], u_data[n_train:]

    if verbose:
        print(f"Train samples: {n_train}, Test samples: {n_samples - n_train}")

    # Build FNO2D model
    model = FNO2D(
        modes=(modes, modes),
        width=width,
        depth=depth,
        out_channels=1,
        activation="gelu",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=RelativeL2Loss(),
        metrics=[RelativeL2Loss(name="rel_l2")],
    )

    history = model.fit(
        a_train,
        u_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(a_test, u_test),
        verbose=verbose,
    )

    test_loss = float(model.evaluate(a_test, u_test, verbose=0)[0])
    if verbose:
        print(f"Final Test Relative L2 Error: {test_loss:.4f}")

    return model, history, test_loss


if __name__ == "__main__":
    run_experiment(epochs=30, verbose=1)

