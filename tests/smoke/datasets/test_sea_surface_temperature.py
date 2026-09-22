import numpy as np
import pytest

from keras_sciml.datasets import (
    SeaSurfaceTemperatureDataset,
    generate_synthetic_sst,
    load_sea_surface_temperature,
)


def test_synthetic_sst_shapes():
    data = generate_synthetic_sst(n_samples=4, resolution=16, n_timesteps=5, seed=42)
    assert data.shape == (4, 16, 16, 5)
    assert data.dtype == np.float32
    assert np.all(np.isfinite(data))
    # Physically plausible SST values (between ~0 deg C and ~35 deg C)
    assert np.all(data > 0.0)
    assert np.all(data < 40.0)


def test_load_sst_pairs():
    x_tr, y_tr, x_te, y_te, meta = load_sea_surface_temperature(
        use_synthetic=True,
        time_history=1,
        time_future=1,
        step_mode="pairs",
        train_ratio=0.8,
        normalize=True,
        seed=42,
    )
    # 50 total samples -> 40 train, 10 test. 6 timesteps with window size 2 -> 5 pairs per sample.
    assert x_tr.shape == (40 * 5, 64, 64, 1)
    assert y_tr.shape == (40 * 5, 64, 64, 1)
    assert x_te.shape == (10 * 5, 64, 64, 1)
    assert y_te.shape == (10 * 5, 64, 64, 1)
    assert meta["normalized"] is True
    # Normalized training set mean should be ~0 and std ~1
    np.testing.assert_allclose(np.mean(x_tr), 0.0, atol=1e-1)
    np.testing.assert_allclose(np.std(x_tr), 1.0, atol=1e-1)


def test_load_sst_window():
    x_tr, y_tr, x_te, y_te, meta = load_sea_surface_temperature(
        use_synthetic=True,
        time_history=2,
        time_future=1,
        step_mode="window",
        train_ratio=0.7,
        normalize=False,
        seed=42,
    )
    # 50 samples -> 35 train, 15 test
    assert x_tr.shape == (35, 64, 64, 2)
    assert y_tr.shape == (35, 64, 64, 1)
    assert x_te.shape == (15, 64, 64, 2)
    assert y_te.shape == (15, 64, 64, 1)


def test_sst_dataset_class():
    ds_train = SeaSurfaceTemperatureDataset(
        use_synthetic=True,
        split="train",
        time_history=1,
        time_future=1,
        train_ratio=0.8,
        normalize=True,
    )
    assert len(ds_train) == 200
    x0, y0 = ds_train[0]
    assert x0.shape == (64, 64, 1)
    assert y0.shape == (64, 64, 1)

    ds_test = SeaSurfaceTemperatureDataset(
        use_synthetic=True,
        split="test",
        time_history=1,
        time_future=1,
        train_ratio=0.8,
        normalize=True,
    )
    assert len(ds_test) == 50
    x_all, y_all = ds_test.as_tuple()
    assert x_all.shape == (50, 64, 64, 1)
    assert y_all.shape == (50, 64, 64, 1)

