import numpy as np
import pytest

from keras_sciml.datasets import (
    DarcyFlowDataset,
    generate_darcy_2d,
    sample_gaussian_random_field_2d,
)


def test_grf_2d_shapes():
    grf = sample_gaussian_random_field_2d(n_samples=5, nx=16, ny=16)
    assert grf.shape == (5, 16, 16)
    assert grf.dtype == np.float32


def test_darcy_2d_shapes():
    # Piecewise permeability
    a_pw, u_pw, x, y = generate_darcy_2d(n_samples=3, resolution=12, field_type="piecewise", seed=42)
    assert a_pw.shape == (3, 12, 12, 1)
    assert u_pw.shape == (3, 12, 12, 1)
    assert x.shape == (12,)
    assert y.shape == (12,)
    assert a_pw.dtype == np.float32
    assert u_pw.dtype == np.float32

    # Permeability positive everywhere
    assert np.all(a_pw > 0.0)
    # Homogeneous Dirichlet boundary condition u = 0 on boundary
    np.testing.assert_allclose(u_pw[:, 0, :, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(u_pw[:, -1, :, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(u_pw[:, :, 0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(u_pw[:, :, -1, 0], 0.0, atol=1e-5)

    # Lognormal field
    a_ln, u_ln, _, _ = generate_darcy_2d(n_samples=2, resolution=10, field_type="lognormal", seed=43)
    assert a_ln.shape == (2, 10, 10, 1)
    assert u_ln.shape == (2, 10, 10, 1)
    assert np.all(a_ln > 0.0)


def test_darcy_dataset_class():
    ds = DarcyFlowDataset(n_samples=4, resolution=10)
    assert len(ds) == 4
    a0, u0 = ds[0]
    assert a0.shape == (10, 10, 1)
    assert u0.shape == (10, 10, 1)
    a_all, u_all = ds.as_tuple()
    assert a_all.shape == (4, 10, 10, 1)
    assert u_all.shape == (4, 10, 10, 1)

