import numpy as np
import pytest

from keras_sciml.datasets import (
    generate_synthetic_sst,
    load_sea_surface_temperature,
)


def test_sst_seed_reproducibility():
    data1 = generate_synthetic_sst(n_samples=5, resolution=16, n_timesteps=4, seed=123)
    data2 = generate_synthetic_sst(n_samples=5, resolution=16, n_timesteps=4, seed=123)
    np.testing.assert_array_equal(data1, data2)


def test_temporal_continuity_and_dynamics():
    # Real physical SST fields evolve continuously in time
    data = generate_synthetic_sst(n_samples=3, resolution=32, n_timesteps=5, seed=42)
    for sample in range(3):
        for t in range(4):
            t_curr = data[sample, :, :, t]
            t_next = data[sample, :, :, t + 1]
            # Must not be identical (actual advection occurring)
            diff = np.max(np.abs(t_next - t_curr))
            assert diff > 0.05

            # High correlation between consecutive days/steps (smooth physical flow)
            corr = np.corrcoef(t_curr.ravel(), t_next.ravel())[0, 1]
            assert corr > 0.85


def test_step_pairs_temporal_alignment():
    x_tr, y_tr, _, _, _ = load_sea_surface_temperature(
        use_synthetic=True,
        time_history=1,
        time_future=1,
        step_mode="pairs",
        train_ratio=1.0,
        normalize=False,
        seed=99,
    )
    # With 50 samples of length 6:
    # Sample i, window t=0 -> x is t=0, y is t=1
    # Sample i, window t=1 -> x is t=1, y is t=2
    # So y from window 0 must match x from window 1 for each sample
    # The pairs are concatenated across time windows:
    # block 0 has shape (50, 64, 64, 1) for (t=0 -> t=1)
    # block 1 has shape (50, 64, 64, 1) for (t=1 -> t=2)
    block_0_y = y_tr[:50]
    block_1_x = x_tr[50:100]
    np.testing.assert_allclose(block_0_y, block_1_x, atol=1e-6)

