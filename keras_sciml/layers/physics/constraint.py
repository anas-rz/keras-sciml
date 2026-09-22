from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.layers import Layer

from keras_sciml.layers.differential.divergence import Divergence


@keras.saving.register_keras_serializable(package="keras_sciml")
class Constraint(Layer):
    """Physical and mathematical constraint layer for physics-informed learning.

    Enforces physical invariants, bounds, conservation laws, or divergence-free conditions
    on neural network predictions.

    Supported constraint types (`constraint_type`):
    - `"bounds"`: Box bounds $u_{\\min} \\le u \\le u_{\\max}$.
    - `"positivity"`: Lower bound $u \\ge u_{\\min}$ (defaults to 0.0).
    - `"conservation"`: Integral or total sum conservation: $\\int_\\Omega u \\, d\\Omega = \\text{target}$ (or zero-mean).
    - `"incompressibility"`: Divergence-free condition on vector fields: $\\nabla \\cdot \\mathbf{v} = 0$.

    Supported enforcement modes (`enforcement`):
    - `"soft"`: Penalizes violations via squared penalty terms added directly to model loss
      using `self.add_loss(weight * penalty)`.
    - `"hard"`: Explicitly projects or clips predictions to strictly satisfy the constraint:
      - For `"bounds"` / `"positivity"`: Uses `ops.clip` or `ops.maximum`.
      - For `"conservation"`: Additive shift projection ensuring integral/mean equals `target` exactly.

    Args:
        constraint_type: str, type of constraint (`"bounds"`, `"positivity"`, `"conservation"`, `"incompressibility"`).
            Defaults to `"bounds"`.
        min_value: float or None, minimum allowed value for bounds/positivity.
        max_value: float or None, maximum allowed value for bounds.
        target: float, target conserved quantity for conservation constraints. Defaults to 0.0.
        reduction: str, reduction for conservation (`"integral"`, `"sum"`, `"mean"`). Defaults to `"integral"`.
        dx: float or sequence of floats, grid spacing for spatial integrals or divergence. Defaults to 1.0.
        axes: int or sequence of ints, spatial axes where constraint is evaluated. If None,
            defaults to all spatial dimensions `range(1, rank - 1)` (or `[1]` for rank 2).
        weight: float, loss weighting factor when `add_loss=True`. Defaults to 1.0.
        add_loss: bool, whether to register constraint violation MSE loss with `self.add_loss`. Defaults to True.
        enforcement: str, `"soft"` (loss penalty) or `"hard"` (exact projection/clipping). Defaults to `"soft"`.
        return_residual: bool, whether to return `(output, violation_residual)` instead of just `output`.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        constraint_type: str = "bounds",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        target: float = 0.0,
        reduction: str = "integral",
        dx: Union[float, Sequence[float]] = 1.0,
        axes: Optional[Union[int, Sequence[int]]] = None,
        weight: float = 1.0,
        add_loss: bool = True,
        enforcement: str = "soft",
        return_residual: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        constraint_type = constraint_type.lower()
        if constraint_type not in ("bounds", "positivity", "conservation", "incompressibility"):
            raise ValueError(
                f"`constraint_type` must be one of ('bounds', 'positivity', 'conservation', 'incompressibility'), "
                f"received: {constraint_type}"
            )

        reduction = reduction.lower()
        if reduction not in ("integral", "sum", "mean"):
            raise ValueError(f"`reduction` must be one of ('integral', 'sum', 'mean'), received: {reduction}")

        enforcement = enforcement.lower()
        if enforcement not in ("soft", "hard"):
            raise ValueError(f"`enforcement` must be one of ('soft', 'hard'), received: {enforcement}")

        if constraint_type == "positivity" and min_value is None:
            min_value = 0.0

        if constraint_type == "incompressibility" and enforcement == "hard":
            raise ValueError(
                "Hard enforcement for incompressibility is not currently supported; use enforcement='soft'."
            )

        self.constraint_type = constraint_type
        self.min_value = min_value
        self.max_value = max_value
        self.target = float(target)
        self.reduction = reduction
        self.dx = dx
        self.axes = [axes] if isinstance(axes, int) else (list(axes) if axes is not None else None)
        self.weight = float(weight)
        self.add_loss_enabled = bool(add_loss)
        self.enforcement = enforcement
        self.return_residual = bool(return_residual)

        if self.constraint_type == "incompressibility":
            self.divergence_layer = Divergence(dx=self.dx, keepdims=True)
        else:
            self.divergence_layer = None

    def _get_target_axes(self, rank: int) -> List[int]:
        if self.axes is not None:
            return [(ax if ax >= 0 else rank + ax) for ax in self.axes]
        if rank >= 3:
            return list(range(1, rank - 1))
        elif rank == 2:
            return [1]
        else:
            return [0]

    def _get_cell_volume(self, target_axes: List[int]) -> float:
        if isinstance(self.dx, (int, float)):
            return float(self.dx) ** len(target_axes)
        elif isinstance(self.dx, (list, tuple)):
            vol = 1.0
            for h in self.dx:
                vol *= float(h)
            return vol
        return 1.0

    def call(self, inputs: Any) -> Any:
        u = ops.cast(inputs, self.compute_dtype)
        rank = len(u.shape)

        if self.constraint_type in ("bounds", "positivity"):
            return self._call_bounds(u)
        elif self.constraint_type == "conservation":
            return self._call_conservation(u, rank)
        elif self.constraint_type == "incompressibility":
            return self._call_incompressibility(u)
        return u

    def _call_bounds(self, u: Any) -> Any:
        res_list = []
        if self.min_value is not None:
            min_val = ops.cast(self.min_value, self.compute_dtype)
            low_viol = ops.relu(ops.subtract(min_val, u))
            res_list.append(low_viol)
        if self.max_value is not None:
            max_val = ops.cast(self.max_value, self.compute_dtype)
            high_viol = ops.relu(ops.subtract(u, max_val))
            res_list.append(high_viol)

        if len(res_list) == 1:
            total_viol = res_list[0]
        elif len(res_list) == 2:
            total_viol = ops.add(res_list[0], res_list[1])
        else:
            total_viol = ops.zeros_like(u)

        if self.add_loss_enabled and len(res_list) > 0:
            loss = ops.cast(self.weight, self.compute_dtype) * ops.mean(ops.square(total_viol))
            self.add_loss(loss)

        if self.enforcement == "hard":
            u_out = u
            if self.min_value is not None and self.max_value is not None:
                u_out = ops.clip(u_out, self.min_value, self.max_value)
            elif self.min_value is not None:
                u_out = ops.maximum(u_out, ops.cast(self.min_value, self.compute_dtype))
            elif self.max_value is not None:
                u_out = ops.minimum(u_out, ops.cast(self.max_value, self.compute_dtype))

            if self.return_residual:
                return u_out, ops.zeros_like(u)
            return u_out

        if self.return_residual:
            return u, total_viol
        return u

    def _call_conservation(self, u: Any, rank: int) -> Any:
        target_axes = self._get_target_axes(rank)
        dV = self._get_cell_volume(target_axes)
        target = ops.cast(self.target, self.compute_dtype)

        if self.reduction == "integral":
            integral = ops.multiply(ops.sum(u, axis=target_axes, keepdims=True), ops.cast(dV, self.compute_dtype))
            residual = ops.subtract(integral, target)
        elif self.reduction == "sum":
            total_sum = ops.sum(u, axis=target_axes, keepdims=True)
            residual = ops.subtract(total_sum, target)
        else:  # mean
            mean_val = ops.mean(u, axis=target_axes, keepdims=True)
            residual = ops.subtract(mean_val, target)

        if self.add_loss_enabled:
            loss = ops.cast(self.weight, self.compute_dtype) * ops.mean(ops.square(residual))
            self.add_loss(loss)

        if self.enforcement == "hard":
            # Compute number of elements across target_axes dynamically
            shape = ops.shape(u)
            n_elements = ops.cast(1, self.compute_dtype)
            for ax in target_axes:
                n_elements = ops.multiply(n_elements, ops.cast(shape[ax], self.compute_dtype))

            if self.reduction == "integral":
                # Total domain volume V = n_elements * dV
                total_vol = ops.multiply(n_elements, ops.cast(dV, self.compute_dtype))
                shift = ops.divide(residual, total_vol)
            elif self.reduction == "sum":
                shift = ops.divide(residual, n_elements)
            else:  # mean
                shift = residual

            u_enforced = ops.subtract(u, shift)
            if self.return_residual:
                return u_enforced, ops.zeros_like(residual)
            return u_enforced

        if self.return_residual:
            return u, residual
        return u

    def _call_incompressibility(self, u: Any) -> Any:
        div = self.divergence_layer(u)
        if self.add_loss_enabled:
            loss = ops.cast(self.weight, self.compute_dtype) * ops.mean(ops.square(div))
            self.add_loss(loss)

        if self.return_residual:
            return u, div
        return u

    def compute_output_shape(self, input_shape: Any) -> Any:
        if self.return_residual:
            if self.constraint_type == "incompressibility":
                div_shape = self.divergence_layer.compute_output_shape(input_shape)
                return input_shape, div_shape
            elif self.constraint_type == "conservation":
                rank = len(input_shape)
                target_axes = self._get_target_axes(rank)
                res_shape = list(input_shape)
                for ax in target_axes:
                    res_shape[ax] = 1
                return input_shape, tuple(res_shape)
            else:
                return input_shape, input_shape
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "constraint_type": self.constraint_type,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "target": self.target,
                "reduction": self.reduction,
                "dx": self.dx,
                "axes": self.axes,
                "weight": self.weight,
                "add_loss": self.add_loss_enabled,
                "enforcement": self.enforcement,
                "return_residual": self.return_residual,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Constraint":
        return cls(**config)


@keras.saving.register_keras_serializable(package="keras_sciml")
class BoundsConstraint(Constraint):
    """Box bounds constraint ($u_{\\min} \\le u \\le u_{\\max}$) convenience layer."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs: Any,
    ):
        kwargs.pop("constraint_type", None)
        super().__init__(
            constraint_type="bounds",
            min_value=min_value,
            max_value=max_value,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="keras_sciml")
class PositivityConstraint(Constraint):
    """Positivity lower-bound constraint ($u \\ge u_{\\min}$) convenience layer."""

    def __init__(
        self,
        min_value: float = 0.0,
        **kwargs: Any,
    ):
        kwargs.pop("constraint_type", None)
        super().__init__(
            constraint_type="positivity",
            min_value=min_value,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="keras_sciml")
class ConservationConstraint(Constraint):
    """Integral / total sum conservation constraint ($\\int_\\Omega u = \\text{target}$) convenience layer."""

    def __init__(
        self,
        target: float = 0.0,
        reduction: str = "integral",
        dx: Union[float, Sequence[float]] = 1.0,
        axes: Optional[Union[int, Sequence[int]]] = None,
        **kwargs: Any,
    ):
        kwargs.pop("constraint_type", None)
        super().__init__(
            constraint_type="conservation",
            target=target,
            reduction=reduction,
            dx=dx,
            axes=axes,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="keras_sciml")
class IncompressibilityConstraint(Constraint):
    """Incompressibility divergence-free constraint ($\\nabla \\cdot \\mathbf{v} = 0$) convenience layer."""

    def __init__(
        self,
        dx: Union[float, Sequence[float]] = 1.0,
        **kwargs: Any,
    ):
        kwargs.pop("constraint_type", None)
        super().__init__(
            constraint_type="incompressibility",
            dx=dx,
            **kwargs,
        )

