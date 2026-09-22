import math
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import initializers, ops
from keras.layers import Layer


def _rfft_1d(x: Any) -> Tuple[Any, Any]:
    """Computes 1D Real FFT along the last axis. Returns (real, imag)."""
    return ops.rfft(x)


def _irfft_1d(real: Any, imag: Any, fft_length: Any) -> Any:
    """Computes 1D Inverse Real FFT along the last axis."""
    return ops.irfft((real, imag), fft_length=fft_length)


def _rfft_2d(x: Any) -> Tuple[Any, Any]:
    """Computes 2D Real FFT along the last two axes (-2, -1).

    Args:
        x: Tensor of shape (..., H, W).

    Returns:
        Tuple (real, imag) of shape (..., H, W // 2 + 1).
    """
    # 1. RFFT along W (last axis)
    r1, i1 = ops.rfft(x)  # shape (..., H, W // 2 + 1)
    # 2. FFT along H (second-to-last axis)
    r1_t = ops.transpose(r1, (0, 1, 3, 2))  # shape (..., W // 2 + 1, H)
    i1_t = ops.transpose(i1, (0, 1, 3, 2))
    r2_t, i2_t = ops.fft((r1_t, i1_t))
    r2 = ops.transpose(r2_t, (0, 1, 3, 2))
    i2 = ops.transpose(i2_t, (0, 1, 3, 2))
    return r2, i2


def _irfft_2d(real: Any, imag: Any, h: Any, w: Any, compute_dtype: str = "float32") -> Any:
    """Computes 2D Inverse Real FFT along the last two axes.

    Args:
        real: Real coefficients of shape (..., H, W // 2 + 1).
        imag: Imaginary coefficients of shape (..., H, W // 2 + 1).
        h: Height dimension size.
        w: Width dimension size.
        compute_dtype: Data type for intermediate computation.

    Returns:
        Reconstructed real tensor of shape (..., H, W).
    """
    # 1. Inverse complex FFT along H
    r_t = ops.transpose(real, (0, 1, 3, 2))
    i_t = ops.transpose(imag, (0, 1, 3, 2))
    # ifft(Z) = conj(fft(conj(Z))) / N
    fc_r, fc_i = ops.fft((r_t, -i_t))
    h_float = ops.cast(h, compute_dtype)
    inv_r = ops.transpose(fc_r / h_float, (0, 1, 3, 2))
    inv_i = ops.transpose(-fc_i / h_float, (0, 1, 3, 2))

    # 2. Inverse real FFT along W
    return ops.irfft((inv_r, inv_i), fft_length=w)


@keras.saving.register_keras_serializable(package="keras_sciml")
class SpectralConv1D(Layer):
    """1D Spectral Convolution layer for Fourier Neural Operators (FNO).

    Computes frequency-domain convolution using the 1D Fast Fourier Transform (FFT).
    Inputs are transformed into Fourier space, truncated to the lowest `modes` frequency
    components, multiplied by learnable complex weights, and transformed back into the
    physical domain via inverse FFT.

    References:
        - Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations",
          ICLR 2021.

    Args:
        filters: int, dimensionality of the output space (number of output channels).
        modes: int, number of Fourier modes to retain.
        kernel_initializer: Initializer for the kernel weights. If None, initialized
            with uniform distribution scaled by `1 / (in_channels * filters)`.
        use_bias: bool, whether to include an additive bias vector. Defaults to False.
        bias_initializer: Initializer for the bias vector.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        filters: int,
        modes: int,
        kernel_initializer: Optional[Union[str, initializers.Initializer]] = None,
        use_bias: bool = False,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if filters <= 0:
            raise ValueError(f"`filters` must be a positive integer, received: {filters}")
        if modes <= 0:
            raise ValueError(f"`modes` must be a positive integer, received: {modes}")

        self.filters = int(filters)
        self.modes = int(modes)
        self.kernel_initializer = kernel_initializer
        self.use_bias = bool(use_bias)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 3:
            raise ValueError(
                f"`SpectralConv1D` requires a 3D input tensor (batch, length, channels). "
                f"Received input_shape: {input_shape}"
            )
        in_channels = input_shape[-1]
        if in_channels is None:
            raise ValueError("The last dimension (channel dimension) must be defined.")

        scale = 1.0 / math.sqrt(float(in_channels * self.filters))
        if self.kernel_initializer is None:
            init_r = initializers.RandomUniform(minval=-scale, maxval=scale)
            init_i = initializers.RandomUniform(minval=-scale, maxval=scale)
        else:
            init_r = initializers.get(self.kernel_initializer)
            init_i = initializers.get(self.kernel_initializer)

        weight_shape = (in_channels, self.filters, self.modes)
        self.kernel_real = self.add_weight(
            shape=weight_shape,
            initializer=init_r,
            trainable=True,
            name="kernel_real",
        )
        self.kernel_imag = self.add_weight(
            shape=weight_shape,
            initializer=init_i,
            trainable=True,
            name="kernel_imag",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs: Any) -> Any:
        x = ops.cast(inputs, self.compute_dtype)
        size_x = ops.shape(x)[1]

        # Transpose to (batch, in_channels, size_x)
        x_t = ops.transpose(x, (0, 2, 1))

        # 1. 1D Real FFT along length
        x_ft_real, x_ft_imag = _rfft_1d(x_t)

        # 2. Truncate to lowest Fourier modes
        x_ft_real_low = x_ft_real[:, :, : self.modes]
        x_ft_imag_low = x_ft_imag[:, :, : self.modes]

        # 3. Complex multiplication: (batch, in_c, modes) x (in_c, out_c, modes) -> (batch, out_c, modes)
        out_ft_real = ops.einsum("bix,iox->box", x_ft_real_low, self.kernel_real) - ops.einsum(
            "bix,iox->box", x_ft_imag_low, self.kernel_imag
        )
        out_ft_imag = ops.einsum("bix,iox->box", x_ft_real_low, self.kernel_imag) + ops.einsum(
            "bix,iox->box", x_ft_imag_low, self.kernel_real
        )

        # 4. Zero-pad remaining frequencies back to (size_x // 2 + 1)
        num_freqs = size_x // 2 + 1
        pad_len = num_freqs - self.modes
        out_ft_real_pad = ops.pad(out_ft_real, [[0, 0], [0, 0], [0, pad_len]])
        out_ft_imag_pad = ops.pad(out_ft_imag, [[0, 0], [0, 0], [0, pad_len]])

        # 5. Inverse 1D Real FFT
        out = _irfft_1d(out_ft_real_pad, out_ft_imag_pad, fft_length=size_x)

        # 6. Transpose back to channels_last: (batch, size_x, out_channels)
        y = ops.transpose(out, (0, 2, 1))

        if self.use_bias and self.bias is not None:
            y = ops.add(y, self.bias)

        return y

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return (input_shape[0], input_shape[1], self.filters)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "modes": self.modes,
                "kernel_initializer": initializers.serialize(self.kernel_initializer)
                if self.kernel_initializer is not None
                else None,
                "use_bias": self.use_bias,
                "bias_initializer": initializers.serialize(self.bias_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpectralConv1D":
        config = config.copy()
        if config.get("kernel_initializer") is not None:
            config["kernel_initializer"] = initializers.deserialize(config["kernel_initializer"])
        if config.get("bias_initializer") is not None:
            config["bias_initializer"] = initializers.deserialize(config["bias_initializer"])
        return cls(**config)


@keras.saving.register_keras_serializable(package="keras_sciml")
class SpectralConv2D(Layer):
    """2D Spectral Convolution layer for Fourier Neural Operators (FNO).

    Computes 2D frequency-domain convolution using the 2D Fast Fourier Transform.
    Fourier modes are truncated in both spatial dimensions, multiplied by learnable
    complex weights, and transformed back into physical space.

    References:
        - Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations",
          ICLR 2021.

    Args:
        filters: int, number of output channels.
        modes: int or Tuple[int, int], number of Fourier modes to retain along (height, width).
        kernel_initializer: Initializer for the kernel weights.
        use_bias: bool, whether to include an additive bias vector. Defaults to False.
        bias_initializer: Initializer for the bias vector.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        filters: int,
        modes: Union[int, Tuple[int, int], List[int]],
        kernel_initializer: Optional[Union[str, initializers.Initializer]] = None,
        use_bias: bool = False,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if filters <= 0:
            raise ValueError(f"`filters` must be a positive integer, received: {filters}")

        if isinstance(modes, int):
            self.modes1 = int(modes)
            self.modes2 = int(modes)
        elif isinstance(modes, (tuple, list)) and len(modes) == 2:
            self.modes1 = int(modes[0])
            self.modes2 = int(modes[1])
        else:
            raise ValueError(
                f"`modes` must be an integer or a tuple/list of two integers, received: {modes}"
            )

        if self.modes1 <= 0 or self.modes2 <= 0:
            raise ValueError(f"`modes` values must be positive, received: ({self.modes1}, {self.modes2})")

        self.filters = int(filters)
        self.kernel_initializer = kernel_initializer
        self.use_bias = bool(use_bias)
        self.bias_initializer = initializers.get(bias_initializer)

    @property
    def modes(self) -> Tuple[int, int]:
        return (self.modes1, self.modes2)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"`SpectralConv2D` requires a 4D input tensor (batch, height, width, channels). "
                f"Received input_shape: {input_shape}"
            )
        in_channels = input_shape[-1]
        if in_channels is None:
            raise ValueError("The last dimension (channel dimension) must be defined.")

        scale = 1.0 / math.sqrt(float(in_channels * self.filters))
        if self.kernel_initializer is None:
            init_r = initializers.RandomUniform(minval=-scale, maxval=scale)
            init_i = initializers.RandomUniform(minval=-scale, maxval=scale)
        else:
            init_r = initializers.get(self.kernel_initializer)
            init_i = initializers.get(self.kernel_initializer)

        weight_shape = (in_channels, self.filters, self.modes1, self.modes2)
        # Weights for top-left modes [:modes1, :modes2]
        self.kernel1_real = self.add_weight(
            shape=weight_shape,
            initializer=init_r,
            trainable=True,
            name="kernel1_real",
        )
        self.kernel1_imag = self.add_weight(
            shape=weight_shape,
            initializer=init_i,
            trainable=True,
            name="kernel1_imag",
        )

        # Weights for bottom-left modes [-modes1:, :modes2]
        self.kernel2_real = self.add_weight(
            shape=weight_shape,
            initializer=init_r,
            trainable=True,
            name="kernel2_real",
        )
        self.kernel2_imag = self.add_weight(
            shape=weight_shape,
            initializer=init_i,
            trainable=True,
            name="kernel2_imag",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs: Any) -> Any:
        x = ops.cast(inputs, self.compute_dtype)
        batch_size = ops.shape(x)[0]
        h = ops.shape(x)[1]
        w = ops.shape(x)[2]

        # Transpose to (batch, in_channels, height, width)
        x_t = ops.transpose(x, (0, 3, 1, 2))

        # 1. 2D Real FFT
        x_ft_real, x_ft_imag = _rfft_2d(x_t)

        # 2. Extract and multiply top-left modes
        x1_real = x_ft_real[:, :, : self.modes1, : self.modes2]
        x1_imag = x_ft_imag[:, :, : self.modes1, : self.modes2]
        out1_real = ops.einsum("bixy,ioxy->boxy", x1_real, self.kernel1_real) - ops.einsum(
            "bixy,ioxy->boxy", x1_imag, self.kernel1_imag
        )
        out1_imag = ops.einsum("bixy,ioxy->boxy", x1_real, self.kernel1_imag) + ops.einsum(
            "bixy,ioxy->boxy", x1_imag, self.kernel1_real
        )

        # 3. Extract and multiply bottom-left modes
        x2_real = x_ft_real[:, :, -self.modes1 :, : self.modes2]
        x2_imag = x_ft_imag[:, :, -self.modes1 :, : self.modes2]
        out2_real = ops.einsum("bixy,ioxy->boxy", x2_real, self.kernel2_real) - ops.einsum(
            "bixy,ioxy->boxy", x2_imag, self.kernel2_imag
        )
        out2_imag = ops.einsum("bixy,ioxy->boxy", x2_real, self.kernel2_imag) + ops.einsum(
            "bixy,ioxy->boxy", x2_imag, self.kernel2_real
        )

        # 4. Assemble full frequency tensor using pad and concatenate
        num_w = w // 2 + 1
        pad_w = num_w - self.modes2
        out1_r_pad = ops.pad(out1_real, [[0, 0], [0, 0], [0, 0], [0, pad_w]])
        out1_i_pad = ops.pad(out1_imag, [[0, 0], [0, 0], [0, 0], [0, pad_w]])

        out2_r_pad = ops.pad(out2_real, [[0, 0], [0, 0], [0, 0], [0, pad_w]])
        out2_i_pad = ops.pad(out2_imag, [[0, 0], [0, 0], [0, 0], [0, pad_w]])

        mid_h = h - 2 * self.modes1
        mid_r = ops.zeros((batch_size, self.filters, mid_h, num_w), dtype=self.compute_dtype)
        mid_i = ops.zeros((batch_size, self.filters, mid_h, num_w), dtype=self.compute_dtype)

        out_ft_real = ops.concatenate([out1_r_pad, mid_r, out2_r_pad], axis=2)
        out_ft_imag = ops.concatenate([out1_i_pad, mid_i, out2_i_pad], axis=2)

        # 5. Inverse 2D Real FFT
        out = _irfft_2d(out_ft_real, out_ft_imag, h, w, compute_dtype=self.compute_dtype)

        # 6. Transpose back to channels_last: (batch, height, width, out_channels)
        y = ops.transpose(out, (0, 2, 3, 1))

        if self.use_bias and self.bias is not None:
            y = ops.add(y, self.bias)

        return y

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "modes": (self.modes1, self.modes2),
                "kernel_initializer": initializers.serialize(self.kernel_initializer)
                if self.kernel_initializer is not None
                else None,
                "use_bias": self.use_bias,
                "bias_initializer": initializers.serialize(self.bias_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpectralConv2D":
        config = config.copy()
        if config.get("kernel_initializer") is not None:
            config["kernel_initializer"] = initializers.deserialize(config["kernel_initializer"])
        if config.get("bias_initializer") is not None:
            config["bias_initializer"] = initializers.deserialize(config["bias_initializer"])
        return cls(**config)


@keras.saving.register_keras_serializable(package="keras_sciml")
class SpectralConv(Layer):
    """Unified Spectral Convolution layer for Fourier Neural Operators.

    Dispatches to `SpectralConv1D` or `SpectralConv2D` based on either the explicit `rank`
    parameter, the structure of `modes` (integer vs tuple), or the rank of the input tensor.

    Args:
        filters: int, number of output channels.
        modes: int or Tuple[int, ...], number of Fourier modes to retain.
        rank: Optional[int], dimensionality of the domain (1 or 2). If None, inferred
            from `modes` or input tensor rank.
        **kwargs: Arguments passed to the underlying `SpectralConv1D` or `SpectralConv2D`.
    """

    def __init__(
        self,
        filters: int,
        modes: Union[int, Tuple[int, ...], List[int]],
        rank: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.raw_modes = modes
        self.rank = rank
        self.extra_kwargs = kwargs

        # If rank is explicitly specified or inferred from modes
        if self.rank == 1 or (self.rank is None and isinstance(modes, int)):
            self.sublayer = SpectralConv1D(
                filters=self.filters,
                modes=modes if isinstance(modes, int) else modes[0],
                **kwargs,
            )
        elif self.rank == 2 or (self.rank is None and isinstance(modes, (tuple, list)) and len(modes) == 2):
            self.sublayer = SpectralConv2D(
                filters=self.filters,
                modes=modes,
                **kwargs,
            )
        else:
            self.sublayer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if self.sublayer is None:
            if len(input_shape) == 3:
                modes1d = self.raw_modes if isinstance(self.raw_modes, int) else self.raw_modes[0]
                self.sublayer = SpectralConv1D(
                    filters=self.filters,
                    modes=modes1d,
                    **self.extra_kwargs,
                )
            elif len(input_shape) == 4:
                self.sublayer = SpectralConv2D(
                    filters=self.filters,
                    modes=self.raw_modes,
                    **self.extra_kwargs,
                )
            else:
                raise ValueError(
                    f"`SpectralConv` supports 1D (shape rank 3) or 2D (shape rank 4) inputs. "
                    f"Received input_shape: {input_shape}"
                )

        self.sublayer.build(input_shape)
        self.built = True

    def call(self, inputs: Any) -> Any:
        return self.sublayer(inputs)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return self.sublayer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "modes": self.raw_modes,
                "rank": self.rank,
            }
        )
        if self.sublayer is not None:
            sub_config = self.sublayer.get_config()
            for key in ["kernel_initializer", "use_bias", "bias_initializer"]:
                if key in sub_config:
                    config[key] = sub_config[key]
        return config
