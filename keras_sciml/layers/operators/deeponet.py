from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import initializers, ops, saving
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="keras_sciml")
class DeepONetCartesianProduct(Layer):
    """Cartesian product merger layer for DeepONet.

    Computes the dot product between branch network outputs (representing input functions)
    and trunk network outputs (representing query coordinate locations) over a Cartesian grid:

        G(u_i)(y_j) = sum_{k=1}^p b_{i,k} * t_{j,k} + b_0

    Supports both unbatched coordinates (shared grid across batch samples) and batched coordinates:
    - If `branch` has shape `(B, p)` and `trunk` has shape `(N, p)`, output has shape `(B, N)`.
    - If `branch` has shape `(B, p)` and `trunk` has shape `(B, N, p)`, output has shape `(B, N)`.
    - If multi-output is used (`branch` has shape `(B, num_outputs, p)` and `trunk` has shape `(N, p)`),
      output has shape `(B, N, num_outputs)`.

    References:
        - Lu et al., "Learning nonlinear operators via DeepONet based on the universal
          approximation theorem of operators", Nature Machine Intelligence 2021.

    Args:
        use_bias: bool, whether to add a trainable scalar or vector bias. Defaults to True.
        bias_initializer: Initializer for the bias parameter. Defaults to "zeros".
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        use_bias: bool = True,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.use_bias = bool(use_bias)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]) -> None:
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"`{self.__class__.__name__}` expects a list or tuple of two input shapes "
                f"[branch_shape, trunk_shape]. Received input_shape: {input_shape}"
            )
        branch_shape, trunk_shape = input_shape[0], input_shape[1]
        if branch_shape[-1] != trunk_shape[-1]:
            raise ValueError(
                f"Branch feature dimension and Trunk feature dimension must match. "
                f"Received branch: {branch_shape[-1]}, trunk: {trunk_shape[-1]}"
            )

        # Check if multi-output branch: shape (B, num_outputs, p)
        if len(branch_shape) == 3 and len(trunk_shape) in (2, 3):
            num_outputs = branch_shape[1]
            bias_shape = (num_outputs,)
        else:
            bias_shape = (1,)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=bias_shape,
                initializer=self.bias_initializer,
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs: Union[List[Any], Tuple[Any, Any]]) -> Any:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"`{self.__class__.__name__}` expects a list of two tensors [branch, trunk]. "
                f"Received: {inputs}"
            )
        branch, trunk = inputs[0], inputs[1]
        branch = ops.cast(branch, self.compute_dtype)
        trunk = ops.cast(trunk, self.compute_dtype)

        branch_shape = ops.shape(branch)
        trunk_shape = ops.shape(trunk)

        # Case 1: Standard 2D branch (B, p) and 2D trunk (N, p)
        if len(branch.shape) == 2 and len(trunk.shape) == 2:
            out = ops.matmul(branch, ops.transpose(trunk))  # shape: (B, N)

        # Case 2: Standard 2D branch (B, p) and 3D batched trunk (B, N, p)
        elif len(branch.shape) == 2 and len(trunk.shape) == 3:
            out = ops.einsum("bp,bnp->bn", branch, trunk)  # shape: (B, N)

        # Case 3: Multi-output branch (B, O, p) and 2D trunk (N, p)
        elif len(branch.shape) == 3 and len(trunk.shape) == 2:
            out = ops.einsum("bop,np->bno", branch, trunk)  # shape: (B, N, O)

        # Case 4: Multi-output branch (B, O, p) and 3D batched trunk (B, N, p)
        elif len(branch.shape) == 3 and len(trunk.shape) == 3:
            out = ops.einsum("bop,bnp->bno", branch, trunk)  # shape: (B, N, O)

        else:
            raise ValueError(
                f"Unsupported rank combination for Cartesian product: "
                f"branch rank {len(branch.shape)}, trunk rank {len(trunk.shape)}"
            )

        if self.use_bias and self.bias is not None:
            out = ops.add(out, self.bias)

        return out

    def compute_output_shape(
        self, input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]
    ) -> Tuple[Optional[int], ...]:
        branch_shape, trunk_shape = input_shape[0], input_shape[1]
        batch = branch_shape[0]

        if len(branch_shape) == 2 and len(trunk_shape) == 2:
            num_points = trunk_shape[0]
            return (batch, num_points)
        elif len(branch_shape) == 2 and len(trunk_shape) == 3:
            num_points = trunk_shape[1]
            return (batch, num_points)
        elif len(branch_shape) == 3 and len(trunk_shape) == 2:
            num_points = trunk_shape[0]
            num_outputs = branch_shape[1]
            return (batch, num_points, num_outputs)
        elif len(branch_shape) == 3 and len(trunk_shape) == 3:
            num_points = trunk_shape[1]
            num_outputs = branch_shape[1]
            return (batch, num_points, num_outputs)
        return (batch, None)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "use_bias": self.use_bias,
                "bias_initializer": initializers.serialize(self.bias_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepONetCartesianProduct":
        config = config.copy()
        if config.get("bias_initializer") is not None:
            config["bias_initializer"] = initializers.deserialize(config["bias_initializer"])
        return cls(**config)


@keras.saving.register_keras_serializable(package="keras_sciml")
class DeepONetPointwise(Layer):
    """Pointwise (aligned) dot-product merger layer for DeepONet.

    Computes the 1-to-1 inner product between paired branch and trunk feature vectors:

        G(u_i)(y_i) = sum_{k=1}^p b_{i,k} * t_{i,k} + b_0

    - If `branch` is `(B, p)` and `trunk` is `(B, p)`, output has shape `(B, 1)`.
    - If `branch` is `(B, N, p)` and `trunk` is `(B, N, p)`, output has shape `(B, N, 1)`.

    Args:
        use_bias: bool, whether to include a trainable bias. Defaults to True.
        bias_initializer: Initializer for the bias parameter. Defaults to "zeros".
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        use_bias: bool = True,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.use_bias = bool(use_bias)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]) -> None:
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"`{self.__class__.__name__}` expects a list or tuple of two input shapes "
                f"[branch_shape, trunk_shape]. Received input_shape: {input_shape}"
            )
        branch_shape, trunk_shape = input_shape[0], input_shape[1]
        if branch_shape[-1] != trunk_shape[-1]:
            raise ValueError(
                f"Branch feature dimension and Trunk feature dimension must match. "
                f"Received branch: {branch_shape[-1]}, trunk: {trunk_shape[-1]}"
            )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(1,),
                initializer=self.bias_initializer,
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs: Union[List[Any], Tuple[Any, Any]]) -> Any:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"`{self.__class__.__name__}` expects a list of two tensors [branch, trunk]. "
                f"Received: {inputs}"
            )
        branch = ops.cast(inputs[0], self.compute_dtype)
        trunk = ops.cast(inputs[1], self.compute_dtype)

        # Pointwise inner product along feature axis
        out = ops.sum(ops.multiply(branch, trunk), axis=-1, keepdims=True)

        if self.use_bias and self.bias is not None:
            out = ops.add(out, self.bias)

        return out

    def compute_output_shape(
        self, input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]
    ) -> Tuple[Optional[int], ...]:
        branch_shape = input_shape[0]
        return tuple(list(branch_shape[:-1]) + [1])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "use_bias": self.use_bias,
                "bias_initializer": initializers.serialize(self.bias_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepONetPointwise":
        config = config.copy()
        if config.get("bias_initializer") is not None:
            config["bias_initializer"] = initializers.deserialize(config["bias_initializer"])
        return cls(**config)


@keras.saving.register_keras_serializable(package="keras_sciml")
class DeepONet(Layer):
    """Deep Operator Network (DeepONet) layer.

    Combines branch and trunk representations to approximate non-linear operators:
        G(u)(y) = sum_{k=1}^p Branch(u)_k * Trunk(y)_k + b_0

    Can be used in two modes:
    1. **With sub-networks**: Pass `branch_net` and `trunk_net`. In this case, `call([u, y])`
       routes the function sensors `u` through `branch_net` and coordinates `y` through `trunk_net`
       before merging them.
    2. **As a direct merger**: Leave `branch_net=None` and `trunk_net=None`. In this case,
       `call([branch_features, trunk_features])` directly merges precomputed feature vectors.

    References:
        - Lu et al., "Learning nonlinear operators via DeepONet based on the universal
          approximation theorem of operators", Nature Machine Intelligence 2021.

    Args:
        branch_net: Optional Keras Layer or Model computing branch features.
        trunk_net: Optional Keras Layer or Model computing trunk features.
        mode: str, either "cartesian" (all pairs of functions and coordinates) or
            "pointwise" (1-to-1 paired functions and coordinates). Defaults to "cartesian".
        use_bias: bool, whether to include a trainable bias. Defaults to True.
        bias_initializer: Initializer for the bias parameter. Defaults to "zeros".
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        branch_net: Optional[Layer] = None,
        trunk_net: Optional[Layer] = None,
        mode: str = "cartesian",
        use_bias: bool = True,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if mode not in ("cartesian", "pointwise"):
            raise ValueError(f"`mode` must be 'cartesian' or 'pointwise', received: {mode}")

        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.mode = mode
        self.use_bias = bool(use_bias)
        self.bias_initializer = initializers.get(bias_initializer)

        if self.mode == "cartesian":
            self.merger = DeepONetCartesianProduct(
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                name="cartesian_merger",
            )
        else:
            self.merger = DeepONetPointwise(
                use_bias=self.use_bias,
                bias_initializer=self.bias_initializer,
                name="pointwise_merger",
            )

    def build(self, input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]) -> None:
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"`{self.__class__.__name__}` expects a list of two input shapes [u_shape, y_shape]. "
                f"Received: {input_shape}"
            )
        u_shape, y_shape = input_shape[0], input_shape[1]

        if self.branch_net is not None:
            if not self.branch_net.built:
                self.branch_net.build(u_shape)
            branch_out_shape = self.branch_net.compute_output_shape(u_shape)
        else:
            branch_out_shape = u_shape

        if self.trunk_net is not None:
            if not self.trunk_net.built:
                self.trunk_net.build(y_shape)
            trunk_out_shape = self.trunk_net.compute_output_shape(y_shape)
        else:
            trunk_out_shape = y_shape

        self.merger.build([branch_out_shape, trunk_out_shape])
        self.built = True

    def call(self, inputs: Union[List[Any], Tuple[Any, Any]]) -> Any:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"`{self.__class__.__name__}` expects a list of two inputs [u, y]. "
                f"Received: {inputs}"
            )
        u, y = inputs[0], inputs[1]

        b = self.branch_net(u) if self.branch_net is not None else u
        t = self.trunk_net(y) if self.trunk_net is not None else y

        return self.merger([b, t])

    def compute_output_shape(
        self, input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]
    ) -> Tuple[Optional[int], ...]:
        u_shape, y_shape = input_shape[0], input_shape[1]
        b_shape = self.branch_net.compute_output_shape(u_shape) if self.branch_net is not None else u_shape
        t_shape = self.trunk_net.compute_output_shape(y_shape) if self.trunk_net is not None else y_shape
        return self.merger.compute_output_shape([b_shape, t_shape])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "branch_net": saving.serialize_keras_object(self.branch_net)
                if self.branch_net is not None
                else None,
                "trunk_net": saving.serialize_keras_object(self.trunk_net)
                if self.trunk_net is not None
                else None,
                "mode": self.mode,
                "use_bias": self.use_bias,
                "bias_initializer": initializers.serialize(self.bias_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepONet":
        config = config.copy()
        if config.get("branch_net") is not None:
            config["branch_net"] = saving.deserialize_keras_object(config["branch_net"])
        if config.get("trunk_net") is not None:
            config["trunk_net"] = saving.deserialize_keras_object(config["trunk_net"])
        if config.get("bias_initializer") is not None:
            config["bias_initializer"] = initializers.deserialize(config["bias_initializer"])
        return cls(**config)

