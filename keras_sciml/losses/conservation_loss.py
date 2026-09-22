from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.losses import Loss

from keras_sciml.layers.differential.divergence import Divergence


@keras.saving.register_keras_serializable(package="keras_sciml")
class ConservationLoss(Loss):
    """Conservation violation loss for physics-informed learning.

    Enforces conservation invariants such as:
    - `"integral"`: Total domain volume/mass integral: $\\int_\\Omega u \\, d\\Omega = \\text{target}$
      or conservation against initial/ground truth state: $\\int_\\Omega \\hat{y} \\, d\\Omega = \\int_\\Omega y \\, d\\Omega$.
    - `"sum"`: Discrete sum conservation: $\\sum_i u_i = \\text{target}$.
    - `"mean"`: Zero-mean or target-mean invariance: $\\bar{u} = \\text{target}$.
    - `"incompressibility"` (or `"divergence"`): Divergence-free condition for vector fields: $\\nabla \\cdot \\mathbf{v} = 0$.

    Args:
        reduction_type: str, type of conservation (`"integral"`, `"sum"`, `"mean"`, `"incompressibility"`).
            Defaults to `"integral"`.
        target: float or None, target conserved value. If None and `y_true` is provided, the conserved quantity
            of `y_pred` is matched to that of `y_true`. Defaults to None.
        dx: float or sequence of floats, spatial grid spacing. Defaults to 1.0.
        axes: int or sequence of ints, spatial axes over which conservation is computed.
            If None, inferred as spatial dimensions (`range(1, rank - 1)` or `(1,)`).
        relative: bool, whether to compute relative conservation violation. Defaults to False.
        loss_type: str, penalty norm (`"mse"` or `"mae"`). Defaults to `"mse"`.
        epsilon: float, numerical stability epsilon. Defaults to 1e-7.
        name: str, loss name. Defaults to `"conservation_loss"`.
        reduction: str, Keras loss reduction mode. Defaults to `"sum_over_batch_size"`.
    """

    def __init__(
        self,
        reduction_type: str = "integral",
        target: Optional[float] = None,
        dx: Union[float, Sequence[float]] = 1.0,
        axes: Optional[Union[int, Sequence[int]]] = None,
        relative: bool = False,
        loss_type: str = "mse",
        epsilon: float = 1e-7,
        name: str = "conservation_loss",
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__(name=name, reduction=reduction)
        reduction_type = reduction_type.lower()
        if reduction_type not in ("integral", "sum", "mean", "incompressibility", "divergence"):
            raise ValueError(
                f"`reduction_type` must be one of ('integral', 'sum', 'mean', 'incompressibility', 'divergence'), "
                f"received: {reduction_type}"
            )

        loss_type = loss_type.lower()
        if loss_type not in ("mse", "mae"):
            raise ValueError(f"`loss_type` must be one of ('mse', 'mae'), received: {loss_type}")

        self.reduction_type = reduction_type
        self.target = float(target) if target is not None else None
        if axes is not None:
            self.axes = (axes,) if isinstance(axes, int) else tuple(axes)
        else:
            self.axes = None
        self.dx = dx
        self.relative = bool(relative)
        self.loss_type = loss_type
        self.epsilon = float(epsilon)

        if self.reduction_type in ("incompressibility", "divergence"):
            self._divergence_layer = Divergence(dx=self.dx)
        else:
            self._divergence_layer = None

    def _get_spatial_axes(self, rank: int) -> Tuple[int, ...]:
        if self.axes is not None:
            return tuple((ax if ax >= 0 else rank + ax) for ax in self.axes)
        if rank >= 3:
            return tuple(range(1, rank - 1))
        elif rank == 2:
            return (1,)
        else:
            return (0,)

    def _get_cell_volume(self, spatial_axes: Tuple[int, ...], rank: int) -> float:
        spatial_dim = len(spatial_axes)
        if isinstance(self.dx, (int, float)):
            return float(self.dx) ** spatial_dim
        elif isinstance(self.dx, (list, tuple)):
            if len(self.dx) == spatial_dim:
                vol = 1.0
                for h in self.dx:
                    vol *= float(h)
                return vol
            elif len(self.dx) == max(1, rank - 2):
                vol = 1.0
                for ax in spatial_axes:
                    vol *= float(self.dx[ax - 1])
                return vol
            else:
                raise ValueError(
                    f"Length of `dx` ({len(self.dx)}) must match spatial axes count ({spatial_dim})."
                )
        else:
            raise TypeError(f"Unsupported type for `dx`: {type(self.dx)}")

    def _compute_quantity(self, u: Any, rank: int, spatial_axes: Tuple[int, ...]) -> Any:
        if self.reduction_type == "integral":
            vol = self._get_cell_volume(spatial_axes, rank)
            return ops.sum(u, axis=spatial_axes) * ops.cast(vol, u.dtype)
        elif self.reduction_type == "sum":
            return ops.sum(u, axis=spatial_axes)
        elif self.reduction_type == "mean":
            return ops.mean(u, axis=spatial_axes)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction_type}")

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_pred = ops.convert_to_tensor(y_pred)
        rank = ops.ndim(y_pred)

        if self.reduction_type in ("incompressibility", "divergence"):
            div = self._divergence_layer(y_pred)
            div_rank = ops.ndim(div)
            reduce_axes = tuple(range(1, div_rank)) if div_rank > 1 else None
            if reduce_axes is not None:
                return (
                    ops.mean(ops.square(div), axis=reduce_axes)
                    if self.loss_type == "mse"
                    else ops.mean(ops.abs(div), axis=reduce_axes)
                )
            else:
                return ops.square(div) if self.loss_type == "mse" else ops.abs(div)

        spatial_axes = self._get_spatial_axes(rank)
        q_pred = self._compute_quantity(y_pred, rank, spatial_axes)

        if self.target is not None:
            target_val = ops.cast(self.target, q_pred.dtype)
        elif y_true is not None:
            y_true = ops.cast(y_true, dtype=y_pred.dtype)
            target_val = self._compute_quantity(y_true, rank, spatial_axes)
        else:
            target_val = ops.zeros_like(q_pred)

        diff = q_pred - target_val
        if self.relative:
            diff = diff / (ops.abs(target_val) + self.epsilon)

        res = ops.square(diff) if self.loss_type == "mse" else ops.abs(diff)

        # If q_pred still has channel dimensions, average across them
        if ops.ndim(res) > 1:
            channel_axes = tuple(range(1, ops.ndim(res)))
            return ops.mean(res, axis=channel_axes)
        return res

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "reduction_type": self.reduction_type,
            "target": self.target,
            "dx": self.dx,
            "axes": self.axes,
            "relative": self.relative,
            "loss_type": self.loss_type,
            "epsilon": self.epsilon,
        }

