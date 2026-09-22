"""Sea Surface Temperature (SST) Spatiotemporal Forecasting with FNO2D.

This example trains a 2D Fourier Neural Operator (FNO2D) to forecast future
ocean surface temperature fields from historical observations using real NOAA
Optimum Interpolation Sea Surface Temperature (OISST) satellite/buoy data.
"""

from typing import Optional, Tuple
import keras
import numpy as np

from keras_sciml.datasets import load_sea_surface_temperature
from keras_sciml.losses import RelativeL2Loss, relative_l2_error
from keras_sciml.models import FNO2D


def run_experiment(
    n_samples: Optional[int] = None,
    resolution: Optional[int] = None,
    time_history: int = 1,
    time_future: int = 1,
    modes: int = 8,
    width: int = 24,
    depth: int = 3,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.005,
    use_synthetic: bool = False,
    verbose: int = 1,
) -> Tuple[keras.Model, keras.callbacks.History, float]:
    """Train FNO2D on Sea Surface Temperature forecasting.

    Args:
        n_samples: Optional sample count override (for quick testing).
        resolution: Optional spatial resolution override (e.g. 16 or 32 for tests).
        time_history: Number of history time steps as input (channels).
        time_future: Number of future time steps to forecast (out_channels).
        modes: Number of Fourier modes in each spatial dimension (x and y).
        width: Latent channel width in spectral convolution layers.
        depth: Number of spectral convolution blocks.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizer.
        use_synthetic: Force use of synthetic physical SST data instead of remote download.
        verbose: Verbosity level for keras.fit.

    Returns:
        `(model, history, test_relative_l2)`
    """
    keras.utils.set_random_seed(42)

    if verbose:
        print("Loading Sea Surface Temperature dataset...")

    x_train, y_train, x_test, y_test, meta = load_sea_surface_temperature(
        download=True,
        n_samples=n_samples,
        resolution=resolution,
        time_history=time_history,
        time_future=time_future,
        step_mode="pairs",
        train_ratio=0.8,
        normalize=True,
        use_synthetic=use_synthetic,
        seed=42,
    )

    if verbose:
        print(f"Data source: {meta['source']}")
        print(f"Train samples: {x_train.shape[0]} with shape {x_train.shape[1:]}")
        print(f"Test samples:  {x_test.shape[0]} with shape {x_test.shape[1:]}")
        print(f"Mean SST: {meta['mean']:.2f} deg C, Std: {meta['std']:.2f} deg C")

    # Build 2D Fourier Neural Operator
    model = FNO2D(
        modes=(modes, modes),
        width=width,
        depth=depth,
        out_channels=time_future,
        activation="gelu",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=RelativeL2Loss(),
        metrics=[RelativeL2Loss(name="rel_l2"), "mae"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Evaluate on test set
    predictions = model.predict(x_test, verbose=0)
    err_tensor = relative_l2_error(y_test, predictions)
    test_rel_l2 = float(np.mean(keras.ops.convert_to_numpy(err_tensor)))

    # Compute unstandardized temperature MAE in degrees Celsius
    std_val = meta["std"]
    y_test_np = keras.ops.convert_to_numpy(y_test)
    preds_np = keras.ops.convert_to_numpy(predictions)
    mae_celsius = float(np.mean(np.abs(y_test_np - preds_np)) * std_val)

    if verbose:
        print(f"\nFinal Test Relative L2 Error: {test_rel_l2:.4f}")
        print(f"Test Mean Absolute Error:    {mae_celsius:.3f} deg C")

    return model, history, test_rel_l2


if __name__ == "__main__":
    run_experiment(epochs=25, verbose=1)
