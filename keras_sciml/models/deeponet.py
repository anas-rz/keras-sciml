from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import layers, ops
from keras.models import Model

from keras_sciml.layers.operators.deeponet import (
    DeepONetCartesianProduct,
    DeepONetPointwise,
)


@keras.saving.register_keras_serializable(package="keras_sciml")
class DeepONet(Model):
    """Deep Operator Network (DeepONet) model for continuous nonlinear operator learning.

    Learns an operator $G: u \\mapsto G(u)$ using the generalized universal approximation theorem:
    $$G(u)(y) = \\sum_{k=1}^p \\text{Branch}(u)_k \\cdot \\text{Trunk}(y)_k + b_0$$

    Args:
        branch_net: Optional Layer or Model for the branch network. If None, built as an MLP.
        trunk_net: Optional Layer or Model for the trunk network. If None, built as an MLP.
        latent_dim: int, number of basis functions / latent dimensions $p$. Defaults to 64.
        mode: str, evaluation mode (`"cartesian"` or `"pointwise"`). Defaults to `"cartesian"`.
        branch_input_dim: Optional[int], input dimension for branch net when auto-building.
        trunk_input_dim: Optional[int], input dimension for trunk net when auto-building.
        hidden_layers: int, number of hidden layers for auto-built branch/trunk MLPs. Defaults to 3.
        hidden_units: int, hidden units per layer for auto-built MLPs. Defaults to 64.
        activation: str, activation function for auto-built MLPs. Defaults to `"tanh"`.
        use_bias: bool, whether to include a trainable scalar output bias $b_0$. Defaults to True.
        name: str, model name. Defaults to `"deeponet"`.
    """

    def __init__(
        self,
        branch_net: Optional[Any] = None,
        trunk_net: Optional[Any] = None,
        latent_dim: int = 64,
        mode: str = "cartesian",
        branch_input_dim: Optional[int] = None,
        trunk_input_dim: Optional[int] = None,
        hidden_layers: int = 3,
        hidden_units: int = 64,
        activation: str = "tanh",
        use_bias: bool = True,
        name: str = "deeponet",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        mode = mode.lower()
        if mode not in ("cartesian", "pointwise"):
            raise ValueError(f"`mode` must be 'cartesian' or 'pointwise', received: {mode}")

        self.latent_dim = int(latent_dim)
        self.mode = mode
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.hidden_layers = int(hidden_layers)
        self.hidden_units = int(hidden_units)
        self.activation_name = activation
        self.use_bias = bool(use_bias)

        # Build Branch Net
        if branch_net is not None:
            self.branch_net = branch_net
        else:
            branch_seq = []
            for _ in range(self.hidden_layers):
                branch_seq.append(layers.Dense(self.hidden_units, activation=self.activation_name))
            branch_seq.append(layers.Dense(self.latent_dim))
            self.branch_net = keras.Sequential(branch_seq, name="branch_net")

        # Build Trunk Net
        if trunk_net is not None:
            self.trunk_net = trunk_net
        else:
            trunk_seq = []
            for _ in range(self.hidden_layers):
                trunk_seq.append(layers.Dense(self.hidden_units, activation=self.activation_name))
            trunk_seq.append(layers.Dense(self.latent_dim))
            self.trunk_net = keras.Sequential(trunk_seq, name="trunk_net")

        if self.mode == "cartesian":
            self.merger = DeepONetCartesianProduct(use_bias=self.use_bias)
        else:
            self.merger = DeepONetPointwise(use_bias=self.use_bias)

    def call(self, inputs: Union[Tuple[Any, Any], List[Any]]) -> Any:
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(f"DeepONet expects inputs as a tuple of (branch_input, trunk_input), got: {type(inputs)}")

        u, y = inputs
        b_feat = self.branch_net(u)
        t_feat = self.trunk_net(y)
        return self.merger([b_feat, t_feat])

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "branch_net": keras.saving.serialize_keras_object(self.branch_net),
            "trunk_net": keras.saving.serialize_keras_object(self.trunk_net),
            "latent_dim": self.latent_dim,
            "mode": self.mode,
            "branch_input_dim": self.branch_input_dim,
            "trunk_input_dim": self.trunk_input_dim,
            "hidden_layers": self.hidden_layers,
            "hidden_units": self.hidden_units,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepONet":
        branch_net_config = config.pop("branch_net", None)
        trunk_net_config = config.pop("trunk_net", None)

        branch_net = keras.saving.deserialize_keras_object(branch_net_config) if branch_net_config else None
        trunk_net = keras.saving.deserialize_keras_object(trunk_net_config) if trunk_net_config else None

        return cls(branch_net=branch_net, trunk_net=trunk_net, **config)

