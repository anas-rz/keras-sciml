"""1D Viscous Burgers' Equation Operator Learning with Fourier Neural Operator (FNO).

This example demonstrates training an FNO1D model to learn the nonlinear time-evolution operator:
    u_0(x) |-> u(x, T)
where u satisfies the viscous Burgers' equation:
    u_t + u u_x = nu u_{xx}
"""

import keras
import numpy as np

from keras_sciml.datasets import BurgersDataset
from keras_sciml.losses import H1Loss, RelativeL2Loss
from keras_sciml.models import FNO1D


def run_experiment(
    n_samples: int = 64,
    nx: int = 32,
    modes: int = 8,
    width: int = 32,
    depth: int = 3,
    t_end: float = 0.5,
    nu: float = 0.02,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 0.01,
    verbose: int = 1,
):
    keras.utils.set_random_seed(42)

    if verbose:
        print(f"Generating 1D Burgers dataset ({n_samples} samples, nx={nx}, nu={nu})...")
    ds = BurgersDataset(mode="pseudospectral", n_samples=n_samples, nx=nx, t_end=t_end, nu=nu, seed=42)
    u0, ut = ds.as_tuple()

    n_train = int(0.8 * n_samples)
    u0_train, ut_train = u0[:n_train], ut[:n_train]
    u0_test, ut_test = u0[n_train:], ut[n_train:]

    dx = 1.0 / nx
    model = FNO1D(
        modes=modes,
        width=width,
        depth=depth,
        out_channels=1,
        activation="gelu",
    )

    # Use H1 Sobolev loss with derivative regularization
    loss_fn = H1Loss(alpha=0.05, dx=dx)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[RelativeL2Loss(name="rel_l2")],
    )

    history = model.fit(
        u0_train,
        ut_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(u0_test, ut_test),
        verbose=verbose,
    )

    test_loss = float(model.evaluate(u0_test, ut_test, verbose=0)[0])
    if verbose:
        print(f"Final Test H1 Loss: {test_loss:.4f}")

    return model, history, test_loss


if __name__ == "__main__":
    run_experiment(epochs=30, verbose=1)

