import numpy as np
import pytest
from keras import ops

from keras_sciml.losses import (
    H1Loss,
    PhysicsLoss,
    RelativeL2Loss,
)


def test_relative_l2_parity_numpy():
    rng = np.random.default_rng(42)
    # Batch of 2D operator fields: (batch=5, nx=16, ny=16, channels=2)
    y_true_np = rng.standard_normal((5, 16, 16, 2)).astype(np.float32)
    y_pred_np = y_true_np + 0.1 * rng.standard_normal((5, 16, 16, 2)).astype(np.float32)

    # Reference NumPy calculation
    diff = y_pred_np - y_true_np
    num = np.linalg.norm(diff.reshape(5, -1), axis=1)
    den = np.linalg.norm(y_true_np.reshape(5, -1), axis=1)
    ref_rel_l2 = num / den

    loss_fn = RelativeL2Loss(reduction="none")
    keras_loss = ops.convert_to_numpy(loss_fn(y_true_np, y_pred_np))
    np.testing.assert_allclose(keras_loss, ref_rel_l2, rtol=1e-4, atol=1e-5)

    # Test squared relative L2
    ref_sq = (num / den) ** 2
    loss_sq = RelativeL2Loss(squared=True, reduction="none")
    keras_sq = ops.convert_to_numpy(loss_sq(y_true_np, y_pred_np))
    np.testing.assert_allclose(keras_sq, ref_sq, rtol=1e-4, atol=1e-5)

    # Test L1 relative error: sum(|diff|) / sum(|y_true|)
    num_l1 = np.sum(np.abs(diff.reshape(5, -1)), axis=1)
    den_l1 = np.sum(np.abs(y_true_np.reshape(5, -1)), axis=1)
    ref_rel_l1 = num_l1 / den_l1
    loss_l1 = RelativeL2Loss(p=1.0, reduction="none")
    keras_l1 = ops.convert_to_numpy(loss_l1(y_true_np, y_pred_np))
    np.testing.assert_allclose(keras_l1, ref_rel_l1, rtol=1e-4, atol=1e-5)


def test_h1_loss_parity_numpy():
    rng = np.random.default_rng(123)
    # 1D field with known spatial step
    batch_size = 4
    nx = 32
    dx = 0.05
    x = np.linspace(0, (nx - 1) * dx, nx, dtype=np.float32)

    y_true_np = np.stack([np.sin(x) for _ in range(batch_size)])[:, :, None]
    y_pred_np = np.stack([np.sin(x) + 0.1 * np.cos(2 * x) for _ in range(batch_size)])[:, :, None]

    diff = y_pred_np - y_true_np

    # Reference central difference gradient for interior
    # (H1Loss uses Gradient layer which does central differences)
    grad_diff = np.zeros_like(diff)
    grad_diff[:, 1:-1, :] = (diff[:, 2:, :] - diff[:, :-2, :]) / (2 * dx)
    grad_diff[:, 0, :] = (diff[:, 1, :] - diff[:, 0, :]) / dx
    grad_diff[:, -1, :] = (diff[:, -1, :] - diff[:, -2, :]) / dx

    alpha = 0.2
    l2_term = np.mean(diff ** 2, axis=(1, 2))
    grad_term = np.mean(grad_diff ** 2, axis=(1, 2))
    ref_h1 = l2_term + alpha * grad_term

    h1 = H1Loss(alpha=alpha, dx=dx, reduction="none")
    keras_h1 = ops.convert_to_numpy(h1(y_true_np, y_pred_np))
    np.testing.assert_allclose(keras_h1, ref_h1, rtol=1e-4, atol=1e-5)


def test_physics_loss_parity():
    rng = np.random.default_rng(99)
    y_true = rng.standard_normal((3, 10, 1)).astype(np.float32)
    y_pred = rng.standard_normal((3, 10, 1)).astype(np.float32)

    # Relative L2 via PhysicsLoss vs direct RelativeL2Loss
    pl = PhysicsLoss(loss_type="relative_l2", reduction="none")
    rl2 = RelativeL2Loss(reduction="none")

    np.testing.assert_allclose(
        ops.convert_to_numpy(pl(y_true, y_pred)),
        ops.convert_to_numpy(rl2(y_true, y_pred)),
        rtol=1e-5,
    )

    # MSE via PhysicsLoss vs keras.losses.MeanSquaredError
    pl_mse = PhysicsLoss(loss_type="mse", reduction="none")
    mse = np.mean((y_pred - y_true) ** 2, axis=(1, 2))
    np.testing.assert_allclose(ops.convert_to_numpy(pl_mse(y_true, y_pred)), mse, rtol=1e-5)

