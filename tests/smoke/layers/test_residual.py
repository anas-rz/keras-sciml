import tempfile
import pytest
import keras
from keras import layers, ops
import numpy as np

from keras_sciml.layers.physics import (
    AdvectionResidual,
    BurgersResidual,
    DiffusionResidual,
    PoissonResidual,
    Residual,
)


def test_instantiation():
    r = Residual(equation="poisson", dx=0.1, forcing=1.0)
    assert r.equation == "poisson"
    assert r.dx == 0.1
    assert r.forcing == 1.0

    diff = DiffusionResidual(diffusivity=0.5, dx=0.2)
    assert diff.equation == "diffusion"
    assert diff.diffusivity == 0.5

    burgers = BurgersResidual(nu=0.05, dx=0.05)
    assert burgers.equation == "burgers"
    assert burgers.nu == 0.05

    adv = AdvectionResidual(velocity=2.0)
    assert adv.equation == "advection"
    assert adv.velocity == 2.0

    with pytest.raises(ValueError, match="equation"):
        Residual(equation="unknown")

    with pytest.raises(ValueError, match="pde_fn"):
        Residual(equation="custom")


def test_shapes_poisson_1d_2d_3d():
    # 1D
    r_1d = PoissonResidual(dx=0.1)
    u_1d = ops.ones((2, 20, 1))
    assert r_1d(u_1d).shape == (2, 20, 1)

    # 2D
    r_2d = PoissonResidual(dx=[0.1, 0.2])
    u_2d = ops.ones((2, 16, 16, 1))
    assert r_2d(u_2d).shape == (2, 16, 16, 1)

    # 3D
    r_3d = PoissonResidual(dx=0.1)
    u_3d = ops.ones((1, 8, 8, 8, 2))
    assert r_3d(u_3d).shape == (1, 8, 8, 8, 2)


def test_shapes_burgers_advection():
    # 1D Burgers
    burg = BurgersResidual(nu=0.01, dx=0.05)
    u = ops.ones((2, 25, 1))
    assert burg(u).shape == (2, 25, 1)

    # 1D Advection
    adv = AdvectionResidual(velocity=1.5, dx=0.05)
    assert adv(u).shape == (2, 25, 1)


def test_return_residual():
    r = PoissonResidual(dx=0.1, return_residual=True)
    u = ops.ones((2, 10, 1))
    out, res = r(u)
    assert out.shape == (2, 10, 1)
    assert res.shape == (2, 10, 1)


def test_dynamic_forcing_input():
    r = PoissonResidual(dx=0.1, return_residual=True)
    u = ops.ones((2, 12, 12, 1))
    f = ops.ones((2, 12, 12, 1)) * 3.0

    out, res = r((u, f))
    assert out.shape == (2, 12, 12, 1)
    assert res.shape == (2, 12, 12, 1)


def test_custom_pde_fn():
    def custom_fn(u):
        return ops.square(u) - 1.0

    r = Residual(equation="custom", pde_fn=custom_fn, return_residual=True)
    u = ops.ones((2, 8, 1)) * 2.0
    out, res = r(u)
    assert out.shape == (2, 8, 1)
    # 2^2 - 1 = 3
    res_np = ops.convert_to_numpy(res)
    np.testing.assert_allclose(res_np, 3.0)


def test_add_loss_in_model():
    inputs = layers.Input(shape=(14, 14, 1))
    outputs = PoissonResidual(dx=0.1, weight=5.0, add_loss=True)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    test_in = ops.ones((2, 14, 14, 1))
    _ = model(test_in)
    assert len(model.losses) > 0


def test_serialization_config():
    r = Residual(
        equation="diffusion",
        dx=[0.1, 0.2],
        diffusivity=2.5,
        forcing=1.0,
        boundary_condition="periodic",
        weight=4.0,
        add_loss=False,
        return_residual=True,
    )
    cfg = r.get_config()
    rec = Residual.from_config(cfg)

    assert rec.equation == "diffusion"
    assert rec.dx == [0.1, 0.2]
    assert rec.diffusivity == 2.5
    assert rec.forcing == 1.0
    assert rec.boundary_condition == "periodic"
    assert rec.weight == 4.0
    assert rec.add_loss_enabled is False
    assert rec.return_residual is True


def test_model_save_and_load():
    inputs = layers.Input(shape=(10, 10, 1))
    outputs = PoissonResidual(dx=0.1)(inputs)
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

