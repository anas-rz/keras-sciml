import tempfile
import pytest
import keras
from keras import layers, ops
import numpy as np

from keras_sciml.layers.physics import (
    BoundaryCondition,
    DirichletBC,
    NeumannBC,
    PeriodicBC,
    RobinBC,
)


def test_instantiation():
    bc = BoundaryCondition(bc_type="dirichlet", value=1.0)
    assert bc.bc_type == "dirichlet"
    assert bc.value == 1.0

    dbc = DirichletBC(value=2.0)
    assert dbc.bc_type == "dirichlet"
    assert dbc.value == 2.0

    nbc = NeumannBC(value=0.0)
    assert nbc.bc_type == "neumann"

    rbc = RobinBC(alpha=2.0, beta=3.0, value=1.0)
    assert rbc.bc_type == "robin"
    assert rbc.alpha == 2.0
    assert rbc.beta == 3.0

    pbc = PeriodicBC()
    assert pbc.bc_type == "periodic"

    with pytest.raises(ValueError, match="bc_type"):
        BoundaryCondition(bc_type="invalid")

    with pytest.raises(ValueError, match="sides"):
        BoundaryCondition(sides="invalid")

    with pytest.raises(ValueError, match="enforcement"):
        BoundaryCondition(enforcement="invalid")

    with pytest.raises(ValueError, match="Hard enforcement"):
        BoundaryCondition(bc_type="neumann", enforcement="hard")


def test_shapes_1d_grid():
    # 1D: (batch, length, 1) -> (batch, length, 1)
    bc = DirichletBC(value=0.0)
    u = ops.ones((2, 20, 1))
    out = bc(u)
    assert out.shape == (2, 20, 1)
    assert bc.compute_output_shape((2, 20, 1)) == (2, 20, 1)


def test_shapes_2d_grid():
    # 2D: (batch, h, w, 1)
    bc = NeumannBC(value=0.0, dx=[0.1, 0.2])
    u = ops.ones((2, 16, 16, 1))
    out = bc(u)
    assert out.shape == (2, 16, 16, 1)
    assert bc.compute_output_shape((2, 16, 16, 1)) == (2, 16, 16, 1)


def test_shapes_3d_grid():
    # 3D: (batch, d, h, w, 2)
    bc = RobinBC(alpha=1.0, beta=1.0, value=0.0, dx=0.1)
    u = ops.ones((1, 8, 8, 8, 2))
    out = bc(u)
    assert out.shape == (1, 8, 8, 8, 2)


def test_return_residual():
    bc = DirichletBC(value=0.0, return_residual=True)
    u = ops.ones((2, 10, 1))
    out, res = bc(u)
    assert out.shape == (2, 10, 1)
    assert res is not None


def test_collocation_inputs():
    bc = DirichletBC(weight=2.0)
    u_b = ops.ones((10, 1))
    g_b = ops.zeros((10, 1))
    out = bc((u_b, g_b))
    assert out.shape == (10, 1)
    assert len(bc.losses) > 0


def test_hard_enforcement_values():
    bc = DirichletBC(value=5.0, enforcement="hard")
    u = ops.zeros((2, 12, 12, 1))
    u_enforced = bc(u)

    u_np = ops.convert_to_numpy(u_enforced)
    # Check boundary slices
    np.testing.assert_allclose(u_np[:, 0, :, :], 5.0)
    np.testing.assert_allclose(u_np[:, -1, :, :], 5.0)
    np.testing.assert_allclose(u_np[:, :, 0, :], 5.0)
    np.testing.assert_allclose(u_np[:, :, -1, :], 5.0)
    # Interior was 0
    np.testing.assert_allclose(u_np[:, 1:-1, 1:-1, :], 0.0)


def test_add_loss_in_model():
    inputs = layers.Input(shape=(16, 16, 1))
    x = layers.Dense(8, activation="relu")(inputs)
    x = layers.Dense(1)(x)
    bc_layer = DirichletBC(value=0.0, weight=5.0, add_loss=True)
    outputs = bc_layer(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    test_in = ops.ones((2, 16, 16, 1))
    _ = model(test_in)
    assert len(model.losses) > 0


def test_serialization_config():
    bc = BoundaryCondition(
        bc_type="robin",
        value=2.5,
        axes=[1, 2],
        sides="lower",
        dx=[0.1, 0.2],
        alpha=1.5,
        beta=0.5,
        weight=2.0,
        add_loss=False,
        enforcement="soft",
        return_residual=True,
    )
    cfg = bc.get_config()
    rec = BoundaryCondition.from_config(cfg)

    assert rec.bc_type == "robin"
    assert rec.value == 2.5
    assert rec.axes == [1, 2]
    assert rec.sides == "lower"
    assert rec.dx == [0.1, 0.2]
    assert rec.alpha == 1.5
    assert rec.beta == 0.5
    assert rec.weight == 2.0
    assert rec.add_loss_enabled is False
    assert rec.enforcement == "soft"
    assert rec.return_residual is True


def test_model_save_and_load():
    inputs = layers.Input(shape=(10, 10, 1))
    outputs = DirichletBC(value=1.0, enforcement="hard")(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    test_input = np.random.uniform(size=(2, 10, 10, 1)).astype("float32")
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

