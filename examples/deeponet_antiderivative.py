"""Deep Operator Network (DeepONet) for Antiderivative Operator Learning.

This example demonstrates training a DeepONet to approximate the nonlinear integration operator:
    G(u)(y) = int_0^y u(s) ds
where u(x) is sampled from a Gaussian process / random function distribution.
"""

import keras
import numpy as np

from keras_sciml.losses import RelativeL2Loss
from keras_sciml.models import DeepONet


def generate_antiderivative_data(
    n_samples: int = 100,
    m_sensors: int = 30,
    n_points: int = 40,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    x_sensors = np.linspace(0.0, 1.0, m_sensors, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, n_points, dtype=np.float32)[:, None]

    # Functions: u(x) = sum_{k=1}^K a_k sin(k pi x) + b_k cos(k pi x)
    K = 4
    k = np.arange(1, K + 1, dtype=np.float32)
    scales = 1.0 / (k ** 1.5)

    a = rng.standard_normal((n_samples, K)).astype(np.float32) * scales
    b = rng.standard_normal((n_samples, K)).astype(np.float32) * scales

    # u at sensor locations: shape (n_samples, m_sensors)
    kx_sensor = np.pi * k[:, None] * x_sensors[None, :]
    u_branch = np.matmul(a, np.sin(kx_sensor)) + np.matmul(b, np.cos(kx_sensor))

    # Analytical antiderivative at y:
    # int_0^y sin(k pi s) ds = [1 - cos(k pi y)] / (k pi)
    # int_0^y cos(k pi s) ds = sin(k pi y) / (k pi)
    ky = np.pi * k[:, None] * y_coords.T  # (K, n_points)
    int_sin = (1.0 - np.cos(ky)) / (np.pi * k[:, None])
    int_cos = np.sin(ky) / (np.pi * k[:, None])

    G_target = np.matmul(a, int_sin) + np.matmul(b, int_cos)  # (n_samples, n_points)
    y_batch = np.tile(y_coords[None, :, :], (n_samples, 1, 1))  # (n_samples, n_points, 1)

    return u_branch, y_batch, G_target


def run_experiment(
    n_samples: int = 64,
    m_sensors: int = 30,
    n_points: int = 40,
    latent_dim: int = 32,
    hidden_units: int = 64,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.01,
    verbose: int = 1,
):
    keras.utils.set_random_seed(42)

    if verbose:
        print(f"Generating antiderivative operator dataset ({n_samples} samples)...")
    u_branch, y_batch, G_target = generate_antiderivative_data(
        n_samples=n_samples, m_sensors=m_sensors, n_points=n_points, seed=42
    )

    n_train = int(0.8 * n_samples)
    u_tr, y_tr, G_tr = u_branch[:n_train], y_batch[:n_train], G_target[:n_train]
    u_te, y_te, G_te = u_branch[n_train:], y_batch[n_train:], G_target[n_train:]

    model = DeepONet(
        latent_dim=latent_dim,
        hidden_layers=3,
        hidden_units=hidden_units,
        activation="tanh",
        mode="cartesian",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=RelativeL2Loss(),
        metrics=[RelativeL2Loss(name="rel_l2")],
    )

    history = model.fit(
        [u_tr, y_tr],
        G_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([u_te, y_te], G_te),
        verbose=verbose,
    )

    test_loss = float(model.evaluate([u_te, y_te], G_te, verbose=0)[0])
    if verbose:
        print(f"Final Test Relative L2 Error: {test_loss:.4f}")

    return model, history, test_loss


if __name__ == "__main__":
    run_experiment(epochs=40, verbose=1)

