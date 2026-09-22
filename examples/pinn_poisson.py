"""Physics-Informed Neural Network (PINN) for 1D Poisson Equation.

Solves the boundary value problem:
    -u''(x) = pi^2 sin(pi * x),  x in [0, 1]
    u(0) = 0,  u(1) = 0
with exact analytical solution u(x) = sin(pi * x).
"""

import keras
from keras import ops
import numpy as np

from keras_sciml.models import PINN


def run_experiment(
    n_collocation: int = 128,
    hidden_layers: int = 3,
    hidden_units: int = 64,
    epochs: int = 80,
    lr: float = 0.01,
    verbose: int = 1,
):
    keras.utils.set_random_seed(42)

    # Collocation grid in [0, 1]
    x_colloc = np.linspace(0.0, 1.0, n_collocation, dtype=np.float32)[:, None]
    u_exact = np.sin(np.pi * x_colloc)
    f_source = (np.pi ** 2) * np.sin(np.pi * x_colloc)

    dx = 1.0 / (n_collocation - 1)

    # Define physics residual function: R(u, x) = -u'' - f
    def poisson_residual(u, x):
        # 3-point central second derivative
        # Shape: (batch, 1)
        u_interior = u[1:-1]
        u_prev = u[:-2]
        u_next = u[2:]
        d2u = (u_next - 2.0 * u_interior + u_prev) / (dx ** 2)
        f = (np.pi ** 2) * ops.sin(np.pi * x[1:-1])
        res = -d2u - f
        # Boundary condition penalties: u(0)=0 and u(1)=0
        bc_res = ops.stack([u[0], u[-1]], axis=0)
        return ops.concatenate([res, 10.0 * bc_res], axis=0)

    model = PINN(
        output_dim=1,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation="tanh",
        use_fourier_features=True,
        fourier_dim=16,
        residual_fn=poisson_residual,
        pde_weight=0.1,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )

    # Supervised + Physics residual training
    history = model.fit(
        x_colloc,
        u_exact,
        epochs=epochs,
        batch_size=n_collocation,
        verbose=verbose,
    )

    preds = model.predict(x_colloc, verbose=0)
    l2_error = np.linalg.norm(preds - u_exact) / np.linalg.norm(u_exact)
    if verbose:
        print(f"Final Relative L2 Error to Exact Solution: {l2_error:.4e}")

    return model, history, l2_error


if __name__ == "__main__":
    run_experiment(epochs=60, verbose=1)

