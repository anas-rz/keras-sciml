import tempfile
import pytest
import keras
from keras import layers, ops
import numpy as np

from keras_sciml.layers.physics import (
    BoundsConstraint,
    ConservationConstraint,
    Constraint,
    IncompressibilityConstraint,
    PositivityConstraint,
)


def test_instantiation():
    c = Constraint(constraint_type="bounds", min_value=0.0, max_value=1.0)
    assert c.constraint_type == "bounds"
    assert c.min_value == 0.0
    assert c.max_value == 1.0

    pos = PositivityConstraint()
    assert pos.constraint_type == "positivity"
    assert pos.min_value == 0.0

    cons = ConservationConstraint(target=5.0, reduction="sum")
    assert cons.constraint_type == "conservation"
    assert cons.target == 5.0
    assert cons.reduction == "sum"

    incomp = IncompressibilityConstraint(dx=0.1)
    assert incomp.constraint_type == "incompressibility"
    assert incomp.divergence_layer is not None

    with pytest.raises(ValueError, match="constraint_type"):
        Constraint(constraint_type="invalid")

    with pytest.raises(ValueError, match="reduction"):
        Constraint(constraint_type="conservation", reduction="invalid")

    with pytest.raises(ValueError, match="enforcement"):
        Constraint(enforcement="invalid")

    with pytest.raises(ValueError, match="Hard enforcement"):
        Constraint(constraint_type="incompressibility", enforcement="hard")


def test_shapes_bounds():
    c = BoundsConstraint(min_value=0.0, max_value=1.0)
    u = ops.ones((2, 16, 16, 1))
    out = c(u)
    assert out.shape == (2, 16, 16, 1)
    assert c.compute_output_shape((2, 16, 16, 1)) == (2, 16, 16, 1)


def test_shapes_positivity():
    pos = PositivityConstraint(enforcement="hard")
    u = ops.ones((2, 10, 2))
    out = pos(u)
    assert out.shape == (2, 10, 2)


def test_shapes_conservation():
    cons = ConservationConstraint(target=1.0, reduction="integral", dx=0.1)
    u = ops.ones((2, 12, 12, 1))
    out = cons(u)
    assert out.shape == (2, 12, 12, 1)


def test_shapes_incompressibility():
    incomp = IncompressibilityConstraint(dx=[0.1, 0.2])
    v = ops.ones((2, 14, 14, 2))
    out = incomp(v)
    assert out.shape == (2, 14, 14, 2)


def test_return_residual():
    c = BoundsConstraint(min_value=0.0, max_value=1.0, return_residual=True)
    u = ops.ones((2, 8, 8, 1)) * 2.0
    out, res = c(u)
    assert out.shape == (2, 8, 8, 1)
    assert res.shape == (2, 8, 8, 1)

    cons = ConservationConstraint(target=0.0, reduction="mean", return_residual=True)
    out_c, res_c = cons(u)
    assert out_c.shape == (2, 8, 8, 1)
    assert res_c.shape == (2, 1, 1, 1)


def test_hard_bounds_clipping():
    c = BoundsConstraint(min_value=-1.0, max_value=2.0, enforcement="hard")
    u = ops.convert_to_tensor([[-5.0, 0.5, 10.0]])
    out = c(u)
    out_np = ops.convert_to_numpy(out)
    np.testing.assert_allclose(out_np, [[-1.0, 0.5, 2.0]])


def test_hard_conservation_shift():
    # Integral of 1 over [0, 1]x[0, 1] with dx=0.1 (10x10 grid, cell area = 0.01)
    # Sum is 100, integral is 100 * 0.01 = 1.0. Target is 4.0.
    dx = 0.1
    c = ConservationConstraint(target=4.0, reduction="integral", dx=dx, enforcement="hard")
    u = ops.ones((1, 10, 10, 1))
    out = c(u)
    out_np = ops.convert_to_numpy(out)

    integral_out = np.sum(out_np) * (dx**2)
    np.testing.assert_allclose(integral_out, 4.0, atol=1e-5)


def test_add_loss_in_model():
    inputs = layers.Input(shape=(10, 10, 2))
    outputs = IncompressibilityConstraint(dx=0.1, weight=10.0, add_loss=True)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    test_in = ops.ones((2, 10, 10, 2))
    _ = model(test_in)
    assert len(model.losses) > 0


def test_serialization_config():
    c = Constraint(
        constraint_type="bounds",
        min_value=-2.0,
        max_value=2.0,
        target=0.0,
        reduction="mean",
        dx=0.05,
        axes=[1],
        weight=3.0,
        add_loss=False,
        enforcement="hard",
        return_residual=True,
    )
    cfg = c.get_config()
    rec = Constraint.from_config(cfg)

    assert rec.constraint_type == "bounds"
    assert rec.min_value == -2.0
    assert rec.max_value == 2.0
    assert rec.reduction == "mean"
    assert rec.dx == 0.05
    assert rec.axes == [1]
    assert rec.weight == 3.0
    assert rec.add_loss_enabled is False
    assert rec.enforcement == "hard"
    assert rec.return_residual is True


def test_model_save_and_load():
    inputs = layers.Input(shape=(8, 8, 1))
    outputs = BoundsConstraint(min_value=0.0, max_value=1.0, enforcement="hard")(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    test_input = np.random.uniform(low=-2.0, high=2.0, size=(2, 8, 8, 1)).astype("float32")
    orig_pred = model(test_input)

    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        model.save(tmp.name)
        loaded = keras.models.load_model(tmp.name)
        loaded_pred = loaded(test_input)

    np.testing.assert_allclose(
        ops.convert_to_numpy(orig_pred),
        ops.convert_to_numpy(loaded_pred),
        atol=1e-6,
    )

