from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import layers, ops
from keras.models import Model

from keras_sciml.layers.operators.spectral_conv import (
    SpectralConv1D,
    SpectralConv2D,
)


@keras.saving.register_keras_serializable(package="keras_sciml")
class FNOBlock1D(layers.Layer):
    """Single 1D Fourier Neural Operator block combining SpectralConv1D and skip connection."""

    def __init__(
        self,
        modes: int,
        width: int,
        activation: str = "gelu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.modes = int(modes)
        self.width = int(width)
        self.activation_name = activation
        self.spectral_conv = SpectralConv1D(filters=self.width, modes=self.modes)
        self.bypass = layers.Dense(self.width)
        self.act = layers.Activation(self.activation_name)

    def call(self, inputs: Any) -> Any:
        return self.act(self.spectral_conv(inputs) + self.bypass(inputs))

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "modes": self.modes,
            "width": self.width,
            "activation": self.activation_name,
        }


@keras.saving.register_keras_serializable(package="keras_sciml")
class FNOBlock2D(layers.Layer):
    """Single 2D Fourier Neural Operator block combining SpectralConv2D and skip connection."""

    def __init__(
        self,
        modes: Union[int, Sequence[int]],
        width: int,
        activation: str = "gelu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if isinstance(modes, int):
            self.modes: Tuple[int, int] = (modes, modes)
        else:
            self.modes = (int(modes[0]), int(modes[1]))
        self.width = int(width)
        self.activation_name = activation
        self.spectral_conv = SpectralConv2D(filters=self.width, modes=self.modes)
        self.bypass = layers.Dense(self.width)
        self.act = layers.Activation(self.activation_name)

    def call(self, inputs: Any) -> Any:
        return self.act(self.spectral_conv(inputs) + self.bypass(inputs))

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "modes": self.modes,
            "width": self.width,
            "activation": self.activation_name,
        }


@keras.saving.register_keras_serializable(package="keras_sciml")
class FNO1D(Model):
    """1D Fourier Neural Operator model.

    Maps 1D functions/fields to 1D functions/fields across arbitrary spatial resolutions.

    Args:
        modes: int, number of Fourier modes to retain. Defaults to 16.
        width: int, latent channel dimensionality. Defaults to 32.
        depth: int, number of spectral convolution blocks. Defaults to 4.
        out_channels: Optional[int], output channel count. If None, matches input channels. Defaults to None.
        projection_dim: int, hidden dimension of the output projection MLP. Defaults to 128.
        activation: str, non-linear activation function. Defaults to `"gelu"`.
        name: str, model name. Defaults to `"fno_1d"`.
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 32,
        depth: int = 4,
        out_channels: Optional[int] = None,
        projection_dim: int = 128,
        activation: str = "gelu",
        name: str = "fno_1d",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.modes = int(modes)
        self.width = int(width)
        self.depth = int(depth)
        self.out_channels = int(out_channels) if out_channels is not None else None
        self.projection_dim = int(projection_dim)
        self.activation_name = activation

        # Encoder (lifting)
        self.lifting = layers.Dense(self.width)

        # Spectral convolution blocks
        self.fno_blocks = [
            FNOBlock1D(modes=self.modes, width=self.width, activation=self.activation_name, name=f"fno_block_{i}")
            for i in range(self.depth)
        ]

        # Decoder (projection)
        self.proj1 = layers.Dense(self.projection_dim, activation=self.activation_name)
        self.proj2: Optional[layers.Dense] = None
        if self.out_channels is not None:
            self.proj2 = layers.Dense(self.out_channels)

    def build(self, input_shape: Any) -> None:
        if self.proj2 is None:
            c = input_shape[-1]
            self.proj2 = layers.Dense(c)
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        x = self.lifting(inputs)
        for block in self.fno_blocks:
            x = block(x)
        x = self.proj1(x)
        return self.proj2(x)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "modes": self.modes,
            "width": self.width,
            "depth": self.depth,
            "out_channels": self.out_channels,
            "projection_dim": self.projection_dim,
            "activation": self.activation_name,
        }


@keras.saving.register_keras_serializable(package="keras_sciml")
class FNO2D(Model):
    """2D Fourier Neural Operator model.

    Maps 2D spatial fields to 2D spatial fields across arbitrary grid resolutions.

    Args:
        modes: int or Sequence[int], Fourier modes along (y, x). Defaults to 12.
        width: int, latent channel dimensionality. Defaults to 32.
        depth: int, number of spectral convolution blocks. Defaults to 4.
        out_channels: Optional[int], output channel count. If None, matches input channels. Defaults to None.
        projection_dim: int, hidden dimension of the output projection MLP. Defaults to 128.
        activation: str, non-linear activation function. Defaults to `"gelu"`.
        name: str, model name. Defaults to `"fno_2d"`.
    """

    def __init__(
        self,
        modes: Union[int, Sequence[int]] = 12,
        width: int = 32,
        depth: int = 4,
        out_channels: Optional[int] = None,
        projection_dim: int = 128,
        activation: str = "gelu",
        name: str = "fno_2d",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        if isinstance(modes, int):
            self.modes: Tuple[int, int] = (modes, modes)
        else:
            self.modes = (int(modes[0]), int(modes[1]))
        self.width = int(width)
        self.depth = int(depth)
        self.out_channels = int(out_channels) if out_channels is not None else None
        self.projection_dim = int(projection_dim)
        self.activation_name = activation

        # Encoder (lifting)
        self.lifting = layers.Dense(self.width)

        # Spectral convolution blocks
        self.fno_blocks = [
            FNOBlock2D(modes=self.modes, width=self.width, activation=self.activation_name, name=f"fno_block_{i}")
            for i in range(self.depth)
        ]

        # Decoder (projection)
        self.proj1 = layers.Dense(self.projection_dim, activation=self.activation_name)
        self.proj2: Optional[layers.Dense] = None
        if self.out_channels is not None:
            self.proj2 = layers.Dense(self.out_channels)

    def build(self, input_shape: Any) -> None:
        if self.proj2 is None:
            c = input_shape[-1]
            self.proj2 = layers.Dense(c)
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        x = self.lifting(inputs)
        for block in self.fno_blocks:
            x = block(x)
        x = self.proj1(x)
        return self.proj2(x)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "modes": self.modes,
            "width": self.width,
            "depth": self.depth,
            "out_channels": self.out_channels,
            "projection_dim": self.projection_dim,
            "activation": self.activation_name,
        }


@keras.saving.register_keras_serializable(package="keras_sciml")
class FNO(Model):
    """High-level Fourier Neural Operator model supporting 1D and 2D spatial dimensions.

    Automatically dispatches to 1D or 2D architecture based on input shape.

    Args:
        modes: int or Sequence[int], number of Fourier modes to retain.
            If int, used for all spatial dimensions. Defaults to 12.
        width: int, latent channel dimensionality. Defaults to 32.
        depth: int, number of spectral convolution blocks. Defaults to 4.
        out_channels: Optional[int], output channel count. If None, matches input channels. Defaults to None.
        projection_dim: int, hidden dimension of output projection MLP. Defaults to 128.
        activation: str, non-linear activation. Defaults to `"gelu"`.
        name: str, model name. Defaults to `"fno"`.
    """

    def __init__(
        self,
        modes: Union[int, Sequence[int]] = 12,
        width: int = 32,
        depth: int = 4,
        out_channels: Optional[int] = None,
        projection_dim: int = 128,
        activation: str = "gelu",
        name: str = "fno",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.modes = modes
        self.width = int(width)
        self.depth = int(depth)
        self.out_channels = int(out_channels) if out_channels is not None else None
        self.projection_dim = int(projection_dim)
        self.activation_name = activation

        self._model: Optional[Model] = None

    def build(self, input_shape: Any) -> None:
        rank = len(input_shape)
        if rank == 3:  # (batch, nx, channels) -> 1D
            m = self.modes if isinstance(self.modes, int) else self.modes[0]
            self._model = FNO1D(
                modes=m,
                width=self.width,
                depth=self.depth,
                out_channels=self.out_channels if self.out_channels is not None else input_shape[-1],
                projection_dim=self.projection_dim,
                activation=self.activation_name,
            )
        elif rank == 4:  # (batch, ny, nx, channels) -> 2D
            self._model = FNO2D(
                modes=self.modes,
                width=self.width,
                depth=self.depth,
                out_channels=self.out_channels if self.out_channels is not None else input_shape[-1],
                projection_dim=self.projection_dim,
                activation=self.activation_name,
            )
        else:
            raise ValueError(f"FNO supports 3D (1D spatial) or 4D (2D spatial) inputs, received shape: {input_shape}")

        self._model.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        return self._model(inputs)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "modes": self.modes,
            "width": self.width,
            "depth": self.depth,
            "out_channels": self.out_channels,
            "projection_dim": self.projection_dim,
            "activation": self.activation_name,
        }

