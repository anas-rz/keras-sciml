import numpy as np
import pytest
import keras
from keras import ops

from keras_sciml.layers.physics import (
    BoundsConstraint,
    ConservationConstraint,
    Constraint,
    IncompressibilityConstraint,
    PositivityConstraint,
)


def test_parity_bounds_soft_residual():
    np.random.seed(42)
    u_np = np.array([[-2.0, -0.5, 0.0, 0.5, 1.5, 2.5]], dtype="float32")
    min_val, max_val = 0.0, 1.0

    # Expected violation in numpy: relu(min - u) + relu(u - max)
    ref_viol = np.maximum(0.0, min_val - u_np) + np.maximum(0.0, u_np - max_val)

    c = BoundsConstraint(min_value=min_val, max_value=max_val, return_residual=True, add_loss=False)
    _, keras_viol = c(ops.convert_to_tensor(u_np))
    keras_viol_np = ops.convert_to_numpy(keras_viol)

    np.testing.assert_allclose(keras_viol_np, ref_viol, atol=1e-6)


def test_parity_positivity_hard():
    np.random.seed(123)
    u_np = np.random.randn(3, 10, 2).astype("float32")
    ref_out = np.maximum(u_np, 0.0)

    pos = PositivityConstraint(enforcement="hard")
    keras_out = ops.convert_to_numpy(pos(ops.convert_to_tensor(u_np)))

    np.testing.assert_allclose(keras_out, ref_out, atol=1e-6)


def test_parity_conservation_integral():
    np.random.seed(999)
    batch, h, w, c = 2, 10, 12, 1
    dx = [0.05, 0.1]
    dV = dx[0] * dx[1]
    u_np = np.random.randn(batch, h, w, c).astype("float32")
    target = 2.5

    # Numpy integral: sum over spatial axes * dV
    integral_np = np.sum(u_np, axis=(1, 2), keepdims=True) * dV
    ref_res = integral_np - target

    # Hard shift in numpy
    total_vol = (h * w) * dV
    shift_np = ref_res / total_vol
    ref_enforced = u_np - shift_np

    # Verify soft residual
    cons_soft = ConservationConstraint(target=target, reduction="integral", dx=dx, return_residual=True, add_loss=False)
    _, keras_res = cons_soft(ops.convert_to_tensor(u_np))
    np.testing.assert_allclose(ops.convert_to_numpy(keras_res), ref_res, atol=1e-5)

    # Verify hard shift
    cons_hard = ConservationConstraint(target=target, reduction="integral", dx=dx, enforcement="hard")
    keras_enforced = ops.convert_to_numpy(cons_hard(ops.convert_to_tensor(u_np)))
    np.testing.assert_allclose(keras_enforced, ref_enforced, atol=1e-5)

    # Verify that the integral of keras_enforced is exactly target
    enforced_integrals = np.sum(keras_enforced, axis=(1, 2), keepdims=True) * dV
    np.testing.assert_allclose(enforced_integrals, target, atol=1e-5)


def test_parity_incompressibility_residual():
    # v = (-y, x) -> div = 0 everywhere
    np.random.seed(777)
    n = 16
    h = 0.1
    coords = np.linspace(-1.0, 1.0, n, dtype="float32")
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    vx = -yy
    vy = xx
    v_np = np.stack([vx, vy], axis=-1)[None, ...]

    incomp = IncompressibilityConstraint(dx=h, return_residual=True, add_loss=False)
    _, res = incomp(ops.convert_to_tensor(v_np))
    res_np = ops.convert_to_numpy(res)

    # Divergence of (-y, x) is 0
    np.testing.assert_allclose(res_np, 0.0, atol=1e-6)

