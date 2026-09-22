import numpy as np
import pytest

from keras_sciml.datasets import (
    PoissonDataset,
    generate_poisson_1d,
    generate_poisson_2d,
)


def test_poisson_1d_shapes():
    f, u, x = generate_poisson_1d(n_samples=10, nx=32, num_modes=4, seed=1)
    assert f.shape == (10, 32, 1)
    assert u.shape == (10, 32, 1)
    assert x.shape == (32,)
    assert f.dtype == np.float32
    assert u.dtype == np.float32

    # Check boundary condition u(0)=0 and u(1)=0
    np.testing.assert_allclose(u[:, 0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(u[:, -1, 0], 0.0, atol=1e-5)


def test_poisson_2d_shapes():
    f, u, x, y = generate_poisson_2d(n_samples=5, nx=16, ny=16, num_modes=3, seed=2)
    assert f.shape == (5, 16, 16, 1)
    assert u.shape == (5, 16, 16, 1)
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert f.dtype == np.float32
    assert u.dtype == np.float32

    # Check boundaries u = 0 on all 4 faces
    np.testing.assert_allclose(u[:, 0, :, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(u[:, -1, :, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(u[:, :, 0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(u[:, :, -1, 0], 0.0, atol=1e-5)


def test_poisson_dataset_class():
    ds_1d = PoissonDataset(dim=1, n_samples=8, resolution=20)
    assert len(ds_1d) == 8
    f0, u0 = ds_1d[0]
    assert f0.shape == (20, 1)
    assert u0.shape == (20, 1)
    f_all, u_all = ds_1d.as_tuple()
    assert f_all.shape == (8, 20, 1)

    ds_2d = PoissonDataset(dim=2, n_samples=4, resolution=(12, 12))
    assert len(ds_2d) == 4
    f2, u2 = ds_2d.as_tuple()
    assert f2.shape == (4, 12, 12, 1)
    assert u2.shape == (4, 12, 12, 1)

