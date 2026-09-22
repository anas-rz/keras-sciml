"""Real NOAA Sea Surface Temperature (SST) dataset loader and generator.

This module provides access to real satellite and in-situ buoy observations of
sea surface temperatures (OISST) in the North Atlantic, as well as physically
consistent synthetic SST trajectories for offline training and validation.
"""

from typing import Optional, Tuple, Union
import os
import urllib.error
import keras
import numpy as np

NOAA_SST_URL = (
    "https://raw.githubusercontent.com/pratiknag/FNO-DSTM-Code/master/datasets/sst_data-6t-sample.npy"
)


def generate_synthetic_sst(
    n_samples: int = 50,
    resolution: int = 64,
    n_timesteps: int = 6,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Generate physically plausible 2D Sea Surface Temperature trajectories.

    Simulates thermal advection, geostrophic eddy circulation, and seasonal
    latitudinal thermal gradients on a 2D domain.

    Args:
        n_samples: Number of trajectory samples.
        resolution: Spatial resolution (ny, nx) = (resolution, resolution).
        n_timesteps: Number of temporal snapshots per trajectory.
        seed: Random seed for reproducibility.

    Returns:
        ndarray of shape `(n_samples, resolution, resolution, n_timesteps)` of float32.
    """
    rng = np.random.default_rng(seed)
    nx = ny = resolution

    # Coordinate grids: y corresponds to latitude [0, 1], x to longitude [0, 1]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Base meridional temperature gradient (colder at high latitude y=1, warmer at equator y=0)
    base_sst = 28.0 - 18.0 * Y  # Range ~ 10 deg C to 28 deg C

    trajectories = np.empty((n_samples, ny, nx, n_timesteps), dtype=np.float32)

    for i in range(n_samples):
        # Sample random thermal eddies (vortices)
        n_eddies = rng.integers(3, 7)
        eddy_centers_x = rng.uniform(0.1, 0.9, size=n_eddies)
        eddy_centers_y = rng.uniform(0.1, 0.9, size=n_eddies)
        eddy_strengths = rng.uniform(-4.0, 4.0, size=n_eddies)
        eddy_radii = rng.uniform(0.08, 0.20, size=n_eddies)

        # Drift velocity (eastward Gulf-stream-like advection)
        drift_u = rng.uniform(0.03, 0.08)
        drift_v = rng.uniform(-0.01, 0.02)

        for t in range(n_timesteps):
            t_flt = float(t)
            # Advected coordinates
            x_adv = (X - drift_u * t_flt) % 1.0
            y_adv = np.clip(Y - drift_v * t_flt, 0.0, 1.0)

            sample_t = 28.0 - 18.0 * y_adv

            # Add eddy thermal signatures
            for k in range(n_eddies):
                cx = (eddy_centers_x[k] + drift_u * t_flt) % 1.0
                cy = np.clip(eddy_centers_y[k] + drift_v * t_flt, 0.0, 1.0)
                dist_sq = (x_adv - cx) ** 2 + (y_adv - cy) ** 2
                sample_t += eddy_strengths[k] * np.exp(-dist_sq / (2.0 * (eddy_radii[k] ** 2)))

            # Add small observational measurement noise (~0.05 C)
            noise = rng.normal(0.0, 0.05, size=(ny, nx)).astype(np.float32)
            trajectories[i, :, :, t] = sample_t + noise

    return trajectories


def load_sea_surface_temperature(
    data_path: Optional[str] = None,
    download: bool = True,
    n_samples: Optional[int] = None,
    resolution: Optional[int] = None,
    time_history: int = 1,
    time_future: int = 1,
    step_mode: str = "pairs",
    train_ratio: float = 0.8,
    normalize: bool = False,
    use_synthetic: bool = False,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load or fetch the NOAA Sea Surface Temperature (SST) dataset.

    Extracts consecutive spatial temperature fields for autoregressive or sequence-to-sequence
    spatiotemporal forecasting with neural operators (e.g. FNO2D).

    Args:
        data_path: Optional path to pre-downloaded `.npy` file. If None and download is True,
            downloads from the official benchmark repository and caches locally.
        download: Whether to attempt downloading if `data_path` is not provided.
        n_samples: Optional sample count override (for quick testing/experimentation).
        resolution: Optional spatial resolution override (subsampling or synthetic resolution).
        time_history: Number of history time steps used as model input.
        time_future: Number of future time steps predicted as target.
        step_mode: "pairs" to extract all sliding window pairs across time, or "window"
            to extract a single (history, future) window per trajectory.
        train_ratio: Fraction of trajectories allocated to training.
        normalize: Whether to z-score standardize using training set statistics.
        use_synthetic: If True, generates synthetic SST data without downloading.
        seed: Random seed for train/test splitting and synthetic generation.

    Returns:
        `(x_train, y_train, x_test, y_test, metadata)` where:
        - `x_train`: ndarray of shape `(n_train, ny, nx, time_history)`
        - `y_train`: ndarray of shape `(n_train, ny, nx, time_future)`
        - `x_test`: ndarray of shape `(n_test, ny, nx, time_history)`
        - `y_test`: ndarray of shape `(n_test, ny, nx, time_future)`
        - `metadata`: dict with `'mean'`, `'std'`, `'source'`, `'raw_shape'`.
    """
    raw_data: Optional[np.ndarray] = None
    source = "synthetic"

    if not use_synthetic:
        if data_path is not None and os.path.isfile(data_path):
            raw_data = np.load(data_path).astype(np.float32)
            source = f"file:{data_path}"
        elif download:
            try:
                cached_path = keras.utils.get_file(
                    fname="sst_data-6t-sample.npy",
                    origin=NOAA_SST_URL,
                    cache_subdir="datasets/noaa_sst",
                )
                raw_data = np.load(cached_path).astype(np.float32)
                source = "noaa_oisst_sample"
            except (urllib.error.URLError, OSError, Exception):
                # Fallback gracefully to physical synthetic generation if offline
                raw_data = None

    if raw_data is None:
        target_n = n_samples if n_samples is not None else 50
        target_res = resolution if resolution is not None else 64
        raw_data = generate_synthetic_sst(n_samples=target_n, resolution=target_res, n_timesteps=6, seed=seed)
        source = "synthetic_fallback"
    else:
        if n_samples is not None:
            raw_data = raw_data[:n_samples]
        if resolution is not None and resolution < raw_data.shape[1]:
            stride = raw_data.shape[1] // resolution
            raw_data = raw_data[:, : resolution * stride : stride, : resolution * stride : stride, :]

    # raw_data has shape (N, ny, nx, T)
    n_samples, ny, nx, total_time = raw_data.shape

    # Split trajectory samples first to prevent temporal data leakage across train/test
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_train_samples = int(train_ratio * n_samples)
    train_indices = indices[:n_train_samples]
    test_indices = indices[n_train_samples:]

    raw_train = raw_data[train_indices]
    raw_test = raw_data[test_indices]

    def _extract_windows(data_subset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sub_n = data_subset.shape[0]
        inputs, targets = [], []

        if step_mode == "pairs":
            # Slide over all possible time windows of size (time_history + time_future)
            window_size = time_history + time_future
            for t_start in range(total_time - window_size + 1):
                t_mid = t_start + time_history
                t_end = t_mid + time_future
                inputs.append(data_subset[:, :, :, t_start:t_mid])
                targets.append(data_subset[:, :, :, t_mid:t_end])
            x_out = np.concatenate(inputs, axis=0)
            y_out = np.concatenate(targets, axis=0)
        elif step_mode == "window":
            # Extract first window
            x_out = data_subset[:, :, :, :time_history]
            y_out = data_subset[:, :, :, time_history : time_history + time_future]
        else:
            raise ValueError(f"Unknown step_mode: {step_mode}. Must be 'pairs' or 'window'.")

        return x_out.astype(np.float32), y_out.astype(np.float32)

    x_train, y_train = _extract_windows(raw_train)
    x_test, y_test = _extract_windows(raw_test)

    # Compute normalization statistics strictly from training input data
    mean_val = float(np.mean(x_train))
    std_val = float(np.std(x_train)) + 1e-7

    if normalize:
        x_train = (x_train - mean_val) / std_val
        y_train = (y_train - mean_val) / std_val
        x_test = (x_test - mean_val) / std_val
        y_test = (y_test - mean_val) / std_val

    metadata = {
        "mean": mean_val,
        "std": std_val,
        "source": source,
        "raw_shape": raw_data.shape,
        "normalized": normalize,
    }

    return x_train, y_train, x_test, y_test, metadata


class SeaSurfaceTemperatureDataset:
    """Convenience dataset class for Sea Surface Temperature (SST) spatiotemporal data."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        download: bool = True,
        n_samples: Optional[int] = None,
        resolution: Optional[int] = None,
        time_history: int = 1,
        time_future: int = 1,
        step_mode: str = "pairs",
        train_ratio: float = 0.8,
        normalize: bool = False,
        split: str = "train",
        use_synthetic: bool = False,
        seed: Optional[int] = 42,
    ):
        x_tr, y_tr, x_te, y_te, meta = load_sea_surface_temperature(
            data_path=data_path,
            download=download,
            n_samples=n_samples,
            resolution=resolution,
            time_history=time_history,
            time_future=time_future,
            step_mode=step_mode,
            train_ratio=train_ratio,
            normalize=normalize,
            use_synthetic=use_synthetic,
            seed=seed,
        )

        self.metadata = meta
        self.mean = meta["mean"]
        self.std = meta["std"]
        self.split = split

        if split == "train":
            self.x, self.y = x_tr, y_tr
        elif split == "test":
            self.x, self.y = x_te, y_te
        elif split == "all":
            self.x = np.concatenate([x_tr, x_te], axis=0)
            self.y = np.concatenate([y_tr, y_te], axis=0)
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'test', or 'all'.")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (inputs, targets) tuple."""
        return self.x, self.y
