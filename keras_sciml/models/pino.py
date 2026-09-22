from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import layers, ops
from keras.models import Model

from keras_sciml.models.fno import FNO


@keras.saving.register_keras_serializable(package="keras_sciml")
class PINO(Model):
    """Physics-Informed Neural Operator (PINO) model.

    Integrates operator learning (FNO) with physics constraints (PDE residuals and boundary conditions),
    enabling both supervised training and unsupervised physics-informed zero-shot generalization.

    Args:
        operator_model: Optional Model used as the neural operator backbone. If None, instantiates an FNO.
        modes: int or Sequence[int], number of Fourier modes for auto-instantiated FNO. Defaults to 12.
        width: int, latent channel width for auto-instantiated FNO. Defaults to 32.
        depth: int, number of spectral convolution blocks. Defaults to 4.
        out_channels: Optional[int], output channels. Defaults to None.
        residual_fn: Optional callable computing PDE residual $R(u, inputs)$. Defaults to None.
        pde_weight: float, loss weight for PDE residual. Defaults to 0.1.
        return_residual: bool, whether to return `(u, residual)` instead of just `u`. Defaults to False.
        name: str, model name. Defaults to `"pino"`.
    """

    def __init__(
        self,
        operator_model: Optional[Model] = None,
        modes: Union[int, Sequence[int]] = 12,
        width: int = 32,
        depth: int = 4,
        out_channels: Optional[int] = None,
        residual_fn: Optional[Callable[[Any, Any], Any]] = None,
        pde_weight: float = 0.1,
        return_residual: bool = False,
        name: str = "pino",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.modes = modes
        self.width = int(width)
        self.depth = int(depth)
        self.out_channels = int(out_channels) if out_channels is not None else None
        self.residual_fn = residual_fn
        self.pde_weight = float(pde_weight)
        self.return_residual = bool(return_residual)

        if operator_model is not None:
            self.operator_model = operator_model
        else:
            self.operator_model = FNO(
                modes=self.modes,
                width=self.width,
                depth=self.depth,
                out_channels=self.out_channels,
            )

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        u = self.operator_model(inputs)

        res = None
        if self.residual_fn is not None:
            res = self.residual_fn(u, inputs)
            if training and self.pde_weight > 0.0:
                pde_loss = self.pde_weight * ops.mean(ops.square(res))
                self.add_loss(pde_loss)

        if self.return_residual:
            return u, res
        return u

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "operator_model": keras.saving.serialize_keras_object(self.operator_model),
            "modes": self.modes,
            "width": self.width,
            "depth": self.depth,
            "out_channels": self.out_channels,
            "pde_weight": self.pde_weight,
            "return_residual": self.return_residual,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PINO":
        op_config = config.pop("operator_model", None)
        operator_model = keras.saving.deserialize_keras_object(op_config) if op_config else None
        return cls(operator_model=operator_model, **config)

