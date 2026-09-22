from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import layers, ops
from keras.models import Model

from keras_sciml.layers.operators.fourier_features import FourierFeatures


@keras.saving.register_keras_serializable(package="keras_sciml")
class PINN(Model):
    """Physics-Informed Neural Network (PINN) model.

    Approximates solutions to partial differential equations $u(x, t)$ with optional
    Fourier feature coordinate embeddings and physical residual attachment.

    Args:
        output_dim: int, number of output fields/components. Defaults to 1.
        hidden_layers: int, number of hidden layers in MLP backbone. Defaults to 4.
        hidden_units: int, hidden units per layer. Defaults to 64.
        activation: str, smooth activation function (`"tanh"`, `"swish"`, `"gelu"`). Defaults to `"tanh"`.
        use_fourier_features: bool, whether to apply Fourier feature coordinate mapping. Defaults to False.
        fourier_dim: int, number of Fourier feature channels when enabled. Defaults to 32.
        fourier_scale: float, Gaussian frequency scaling factor for Fourier features. Defaults to 1.0.
        residual_fn: Optional callable computing PDE residual $R(u, x)$ during training. Defaults to None.
        pde_weight: float, loss weight for PDE residual. Defaults to 1.0.
        name: str, model name. Defaults to `"pinn"`.
    """

    def __init__(
        self,
        output_dim: int = 1,
        hidden_layers: int = 4,
        hidden_units: int = 64,
        activation: str = "tanh",
        use_fourier_features: bool = False,
        fourier_dim: int = 32,
        fourier_scale: float = 1.0,
        residual_fn: Optional[Callable[[Any], Any]] = None,
        pde_weight: float = 1.0,
        name: str = "pinn",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.output_dim = int(output_dim)
        self.hidden_layers = int(hidden_layers)
        self.hidden_units = int(hidden_units)
        self.activation_name = activation
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_dim = int(fourier_dim)
        self.fourier_scale = float(fourier_scale)
        self.residual_fn = residual_fn
        self.pde_weight = float(pde_weight)

        # Coordinate embedding
        if self.use_fourier_features:
            self.embed = FourierFeatures(output_dim=self.fourier_dim, scale=self.fourier_scale)
        else:
            self.embed = None

        # Backbone layers
        self.dense_layers = [
            layers.Dense(self.hidden_units, activation=self.activation_name, name=f"dense_{i}")
            for i in range(self.hidden_layers)
        ]
        self.out_layer = layers.Dense(self.output_dim, name="output_layer")

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        x = inputs
        if self.embed is not None:
            x = self.embed(x)
        for layer in self.dense_layers:
            x = layer(x)
        u = self.out_layer(x)

        if training and self.residual_fn is not None and self.pde_weight > 0.0:
            res = self.residual_fn(u, inputs)
            pde_loss = self.pde_weight * ops.mean(ops.square(res))
            self.add_loss(pde_loss)

        return u

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
            "hidden_units": self.hidden_units,
            "activation": self.activation_name,
            "use_fourier_features": self.use_fourier_features,
            "fourier_dim": self.fourier_dim,
            "fourier_scale": self.fourier_scale,
            "pde_weight": self.pde_weight,
        }

