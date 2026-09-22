import numpy as np
import pytest

from keras_sciml.datasets import (
    BurgersDataset,
    generate_burgers_1d,
    generate_burgers_analytical_shock,
)


def test_burgers_analytical_shock_shapes():
    u0, ut, x = generate_burgers_analytical_shock(n_samples=6, nx=32, t=0.5, nu=0.05, seed=10)
    assert u0.shape == (6, 32, 1)
    assert ut.shape == (6, 32, 1)
    assert x.shape == (32,)
    assert u0.dtype == np.float32
    assert ut.dtype == np.float32


def test_burgers_pseudospectral_shapes():
    u0, ut, x = generate_burgers_1d(n_samples=4, nx=32, t_end=0.2, dt=2e-3, nu=0.02, seed=11)
    assert u0.shape == (4, 32, 1)
    assert ut.shape == (4, 32, 1)
    assert x.shape == (32,)
    assert u0.dtype == np.float32
    assert ut.dtype == np.float32


def test_burgers_dataset_class():
    ds_rk4 = BurgersDataset(mode="pseudospectral", n_samples=4, nx=16, t_end=0.1)
    assert len(ds_rk4) == 4
    u0, ut = ds_rk4.as_tuple()
    assert u0.shape == (4, 16, 1)
    assert ut.shape == (4, 16, 1)

    ds_shock = BurgersDataset(mode="shock", n_samples=5, nx=24, t_end=0.2)
    assert len(ds_shock) == 5
    u0_s, ut_s = ds_shock[0]
    assert u0_s.shape == (24, 1)
    assert ut_s.shape == (24, 1)

