import math
from typing import Any, Dict, Optional, Union

import keras
from keras import initializers, ops
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="keras_sciml")
class FourierFeatures(Layer):
    """Random or learnable Fourier Feature Mapping layer.

    Projects coordinates into a higher-dimensional sinusoidal representation:
        gamma(x) = [cos(2 * pi * x @ B), sin(2 * pi * x @ B)]

    This mapping mitigates spectral bias in neural networks, allowing coordinate-based
    models (e.g. PINNs, Neural Radiance Fields, Neural Operators) to learn high-frequency
    functions in low-dimensional domains.

    References:
        - Tancik et al., "Fourier Features Let Networks Learn High Frequency
          Functions in Low Dimensional Domains", NeurIPS 2020.
        - Rahimi & Recht, "Random Features for Large-Scale Kernel Machines",
          NeurIPS 2007.

    Args:
        output_dim: int, total number of sinusoidal features. Must be an even positive integer.
            If `include_input=True`, the final output dimension will be `output_dim + in_features`.
        scale: float, standard deviation of the Gaussian frequency matrix initialization.
            Defaults to 10.0.
        trainable_frequencies: bool, whether the frequency matrix `B` is a trainable weight.
            Defaults to False.
        include_input: bool, whether to concatenate the original input coordinates to the output.
            Defaults to False.
        frequency_initializer: Initializer or string identifier for the frequency matrix.
            If None or "normal", `keras.initializers.RandomNormal(stddev=scale)` is used.
        two_pi: bool, whether to multiply coordinates by `2 * pi` before sinusoidal projection.
            Defaults to True.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        output_dim: int,
        scale: float = 10.0,
        trainable_frequencies: bool = False,
        include_input: bool = False,
        frequency_initializer: Optional[Union[str, initializers.Initializer]] = None,
        two_pi: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if output_dim <= 0 or output_dim % 2 != 0:
            raise ValueError(
                f"`output_dim` must be an even positive integer, received: {output_dim}"
            )
        self.output_dim = int(output_dim)
        self.scale = float(scale)
        self.trainable_frequencies = bool(trainable_frequencies)
        self.include_input = bool(include_input)
        self.two_pi = bool(two_pi)

        if frequency_initializer is None or frequency_initializer == "normal":
            self.frequency_initializer = initializers.RandomNormal(
                mean=0.0, stddev=self.scale
            )
        else:
            self.frequency_initializer = initializers.get(frequency_initializer)

    def build(self, input_shape: Any) -> None:
        in_features = input_shape[-1]
        if in_features is None:
            raise ValueError(
                f"The last dimension of inputs to `{self.__class__.__name__}` "
                "must be defined. Received input_shape: {input_shape}"
            )

        num_frequencies = self.output_dim // 2
        self.kernel = self.add_weight(
            shape=(in_features, num_frequencies),
            initializer=self.frequency_initializer,
            trainable=self.trainable_frequencies,
            name="frequencies",
        )
        self.built = True

    def call(self, inputs: Any) -> Any:
        x = ops.cast(inputs, self.compute_dtype)
        if self.two_pi:
            x_proj = ops.multiply(x, 2.0 * math.pi)
        else:
            x_proj = x

        angles = ops.matmul(x_proj, self.kernel)
        features = ops.concatenate([ops.cos(angles), ops.sin(angles)], axis=-1)

        if self.include_input:
            return ops.concatenate([x, features], axis=-1)
        return features

    def compute_output_shape(self, input_shape: Any) -> Any:
        in_shape = list(input_shape)
        in_features = in_shape[-1]
        total_dim = self.output_dim
        if self.include_input and in_features is not None:
            total_dim += in_features
        return tuple(in_shape[:-1] + [total_dim])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "scale": self.scale,
                "trainable_frequencies": self.trainable_frequencies,
                "include_input": self.include_input,
                "two_pi": self.two_pi,
                "frequency_initializer": initializers.serialize(
                    self.frequency_initializer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FourierFeatures":
        config = config.copy()
        if "frequency_initializer" in config:
            config["frequency_initializer"] = initializers.deserialize(
                config["frequency_initializer"]
            )
        return cls(**config)

