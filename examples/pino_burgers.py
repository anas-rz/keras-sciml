"""Physics-Informed Neural Operator (PINO) for 1D Viscous Burgers' Equation.

This example demonstrates training a PINO model combining operator data loss with
unsupervised PDE residual regularization:
    R(u) = u_t + u u_x - nu u_{xx}
"""

import keras
from keras import ops
import numpy as np

from keras_sciml.datasets import BurgersDataset
from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import PINO


def run_experiment(
    n_samples: int = 40,
    nx: int = 32,
    modes: int = 8,
    width: int = 24,
    depth: int = 2,
    t_end: float = 0.5,
    nu: float = 0.05,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 0.01,
    verbose: int = 1,
):
    keras.utils.set_random_seed(42)

    if verbose:
        print(f"Generating Burgers dataset ({n_samples} samples, nx={nx})...")
    ds = BurgersDataset(mode="pseudospectral", n_samples=n_samples, nx=nx, t_end=t_end, nu=nu, seed=42)
    u0, ut = ds.as_tuple()

    dx = 1.0 / nx

    # Physics residual: finite difference approximation of steady or final state
    def burgers_spatial_residual(u, inputs):
        # inputs is initial condition u0
        # Compute spatial derivatives at t_end
        u_interior = u[:, 1:-1, :]
        u_prev = u[:, :-2, :]
        u_next = u[:, 2:, :]
        u_x = (u_next - u_prev) / (2.0 * dx)
        u_xx = (u_next - 2.0 * u_interior + u_prev) / (dx ** 2)
        # Approximate time derivative from initial to final state
        u_t = (u_interior - inputs[:, 1:-1, :]) / t_end
        return u_t + u_interior * u_x - nu * u_xx

    model = PINO(
        modes=modes,
        width=width,
        depth=depth,
        out_channels=1,
        residual_fn=burgers_spatial_residual,
        pde_weight=0.01,
    )

    loss_fn = RelativeL2Loss()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[RelativeL2Loss(name="rel_l2")],
    )

    history = model.fit(
        u0,
        ut,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    final_loss = history.history["loss"][-1]
    if verbose:
        print(f"Final Total Loss: {final_loss:.4f}")

    return model, history, final_loss


if __name__ == "__main__":
    run_experiment(epochs=30, verbose=1)

