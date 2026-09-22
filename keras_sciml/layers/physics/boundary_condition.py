from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.layers import Layer


def _make_slice(rank: int, axis: int, sl: slice) -> Tuple[slice, ...]:
    """Helper to create a slicing tuple for a specific axis."""
    slices = [slice(None)] * rank
    slices[axis] = sl
    return tuple(slices)


@keras.saving.register_keras_serializable(package="keras_sciml")
class BoundaryCondition(Layer):
    """Boundary condition layer for physics-informed learning and neural operators.

    Supports both soft penalty enforcement (via `self.add_loss`) and hard enforcement.
    Handles both spatial grid fields (e.g., FNO, PINO, CNN tensors) and pointwise collocation points.

    Supported boundary types (`bc_type`):
    - `"dirichlet"`: Prescribes values on the boundary: $u = g$.
    - `"neumann"`: Prescribes normal derivatives: $\\frac{\\partial u}{\\partial n} = g$.
    - `"robin"`: Linear combination of value and normal derivative: $\\alpha u + \\beta \\frac{\\partial u}{\\partial n} = g$.
    - `"periodic"`: Enforces equality across opposing boundaries: $u|_{\\text{lower}} = u|_{\\text{upper}}$.

    Supported enforcement modes (`enforcement`):
    - `"soft"`: Computes boundary residual $R_{BC} = \\mathcal{B}(u) - g$. If `add_loss=True`, adds
      $\\text{weight} \\times \\text{mean}(R_{BC}^2)$ to the layer loss.
    - `"hard"`: Explicitly replaces boundary slices of spatial grid tensors with prescribed boundary values $g$
      using differentiable concatenation (Dirichlet only).

    Supported sides (`sides`):
    - `"both"`: Evaluates / enforces conditions on both lower (index 0) and upper (index -1) faces.
    - `"lower"`: Only on lower boundary face (index 0).
    - `"upper"`: Only on upper boundary face (index -1).

    Args:
        bc_type: str, type of boundary condition (`"dirichlet"`, `"neumann"`, `"robin"`, `"periodic"`).
            Defaults to `"dirichlet"`.
        value: float or tensor, prescribed boundary value $g$. Defaults to 0.0 (homogeneous BC).
        axes: int or sequence of ints, spatial axes where the condition is applied.
            If `None`, defaults to all spatial axes `range(1, rank - 1)` for grid tensors.
        sides: str, which boundary face(s) to enforce (`"both"`, `"lower"`, `"upper"`).
            Defaults to `"both"`.
        dx: float or sequence of floats, grid spacing along each spatial axis (used for normal derivatives
            in Neumann and Robin conditions). Defaults to 1.0.
        alpha: float, Robin boundary value coefficient. Defaults to 1.0.
        beta: float, Robin boundary normal derivative coefficient. Defaults to 1.0.
        weight: float, loss weighting factor when `add_loss=True`. Defaults to 1.0.
        add_loss: bool, whether to register the boundary MSE loss with `self.add_loss`. Defaults to True.
        enforcement: str, `"soft"` (penalty loss) or `"hard"` (exact boundary replacement). Defaults to `"soft"`.
        return_residual: bool, whether to return `(output, residual)` instead of just `output`.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        bc_type: str = "dirichlet",
        value: Union[float, Any] = 0.0,
        axes: Optional[Union[int, Sequence[int]]] = None,
        sides: str = "both",
        dx: Union[float, Sequence[float]] = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        weight: float = 1.0,
        add_loss: bool = True,
        enforcement: str = "soft",
        return_residual: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        bc_type = bc_type.lower()
        if bc_type not in ("dirichlet", "neumann", "robin", "periodic"):
            raise ValueError(
                f"`bc_type` must be one of ('dirichlet', 'neumann', 'robin', 'periodic'), received: {bc_type}"
            )

        sides = sides.lower()
        if sides not in ("both", "lower", "upper"):
            raise ValueError(f"`sides` must be one of ('both', 'lower', 'upper'), received: {sides}")

        enforcement = enforcement.lower()
        if enforcement not in ("soft", "hard"):
            raise ValueError(f"`enforcement` must be one of ('soft', 'hard'), received: {enforcement}")

        if enforcement == "hard" and bc_type != "dirichlet":
            raise ValueError(
                f"Hard enforcement is currently only supported for 'dirichlet' boundary conditions, "
                f"received: bc_type={bc_type}"
            )

        self.bc_type = bc_type
        self.value = value
        self.axes = [axes] if isinstance(axes, int) else (list(axes) if axes is not None else None)
        self.sides = sides
        self.dx = dx
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.weight = float(weight)
        self.add_loss_enabled = bool(add_loss)
        self.enforcement = enforcement
        self.return_residual = bool(return_residual)

    def _get_target_axes(self, rank: int) -> List[int]:
        if self.axes is not None:
            return [(ax if ax >= 0 else rank + ax) for ax in self.axes]
        if rank >= 3:
            # All spatial axes: between batch (0) and channel (rank - 1)
            return list(range(1, rank - 1))
        elif rank == 2:
            # 1D field or collocation points
            return [1]
        else:
            return [0]

    def _get_spacings(self, target_axes: List[int], rank: int) -> Dict[int, float]:
        spatial_dim_count = max(1, rank - 2)
        if isinstance(self.dx, (int, float)):
            return {ax: float(self.dx) for ax in target_axes}
        elif isinstance(self.dx, (list, tuple)):
            if len(self.dx) == len(target_axes):
                return {ax: float(h) for ax, h in zip(target_axes, self.dx)}
            elif len(self.dx) == spatial_dim_count:
                # Map spatial dim index (axis - 1) to spacing
                return {ax: float(self.dx[ax - 1]) for ax in target_axes}
            else:
                raise ValueError(
                    f"Length of `dx` ({len(self.dx)}) must match number of target axes ({len(target_axes)}) "
                    f"or total spatial dimensions ({spatial_dim_count})."
                )
        else:
            raise TypeError(f"Unsupported type for `dx`: {type(self.dx)}")

    def call(self, inputs: Any) -> Any:
        # Check if inputs is a collocation tuple: (u_b, g_b) or (u_b, du_dn_b, g_b)
        if isinstance(inputs, (tuple, list)):
            return self._call_collocation(inputs)

        u = ops.cast(inputs, self.compute_dtype)
        rank = len(u.shape)
        target_axes = self._get_target_axes(rank)
        spacings = self._get_spacings(target_axes, rank)

        # Handle hard enforcement
        if self.enforcement == "hard":
            u_enforced = u
            for axis in target_axes:
                u_enforced = self._apply_hard_dirichlet(u_enforced, axis)

            if self.return_residual:
                # In hard enforcement, residual is identically 0
                res = ops.zeros_like(u)
                return u_enforced, res
            return u_enforced

        # Soft enforcement: compute boundary residuals
        residuals = []
        for axis in target_axes:
            h = spacings[axis]
            axis_residuals = self._compute_axis_residuals(u, axis=axis, h=h)
            residuals.extend(axis_residuals)

        # Aggregate total boundary MSE loss
        if residuals:
            stacked_res_losses = [ops.mean(ops.square(r)) for r in residuals]
            total_b_loss = stacked_res_losses[0]
            for r_loss in stacked_res_losses[1:]:
                total_b_loss = ops.add(total_b_loss, r_loss)

            scaled_loss = ops.cast(self.weight, self.compute_dtype) * total_b_loss
            if self.add_loss_enabled:
                self.add_loss(scaled_loss)

            combined_residual = residuals[0] if len(residuals) == 1 else ops.concatenate(
                [ops.reshape(r, (-1,)) for r in residuals], axis=0
            )
        else:
            combined_residual = ops.zeros((1,), dtype=self.compute_dtype)

        if self.return_residual:
            return u, combined_residual
        return u

    def _call_collocation(self, inputs: Sequence[Any]) -> Any:
        """Evaluate boundary condition for point-cloud / collocation pairs."""
        if len(inputs) == 2:
            u_b, g_b = inputs
            u_b = ops.cast(u_b, self.compute_dtype)
            g_b = ops.cast(g_b, self.compute_dtype)
            res = ops.subtract(u_b, g_b)
        elif len(inputs) == 3:
            u_b, du_dn_b, g_b = inputs
            u_b = ops.cast(u_b, self.compute_dtype)
            du_dn_b = ops.cast(du_dn_b, self.compute_dtype)
            g_b = ops.cast(g_b, self.compute_dtype)
            if self.bc_type == "robin":
                res = ops.subtract(
                    ops.add(
                        ops.cast(self.alpha, self.compute_dtype) * u_b,
                        ops.cast(self.beta, self.compute_dtype) * du_dn_b,
                    ),
                    g_b,
                )
            elif self.bc_type == "neumann":
                res = ops.subtract(du_dn_b, g_b)
            else:
                res = ops.subtract(u_b, g_b)
        else:
            raise ValueError(f"Expected 2 or 3 tensors in collocation input tuple, got {len(inputs)}")

        if self.add_loss_enabled:
            loss = ops.cast(self.weight, self.compute_dtype) * ops.mean(ops.square(res))
            self.add_loss(loss)

        if self.return_residual:
            return inputs[0], res
        return inputs[0]

    def _compute_axis_residuals(self, u: Any, axis: int, h: float) -> List[Any]:
        rank = len(u.shape)
        residuals = []

        # Target value g
        g = ops.cast(self.value, self.compute_dtype) if not isinstance(self.value, (int, float)) else float(self.value)

        # Lower face: index 0
        if self.sides in ("both", "lower"):
            u_low = u[_make_slice(rank, axis, slice(0, 1))]
            if self.bc_type == "dirichlet":
                r_low = ops.subtract(u_low, g)
            elif self.bc_type == "neumann":
                # Outward normal is -e_axis: du/dn = (u_0 - u_1) / h
                u_next = u[_make_slice(rank, axis, slice(1, 2))]
                du_dn_low = ops.divide(ops.subtract(u_low, u_next), h)
                r_low = ops.subtract(du_dn_low, g)
            elif self.bc_type == "robin":
                u_next = u[_make_slice(rank, axis, slice(1, 2))]
                du_dn_low = ops.divide(ops.subtract(u_low, u_next), h)
                robin_term = ops.add(
                    ops.cast(self.alpha, self.compute_dtype) * u_low,
                    ops.cast(self.beta, self.compute_dtype) * du_dn_low,
                )
                r_low = ops.subtract(robin_term, g)
            elif self.bc_type == "periodic":
                u_high = u[_make_slice(rank, axis, slice(-1, None))]
                r_low = ops.subtract(u_low, u_high)
            residuals.append(r_low)

        # Upper face: index -1
        if self.sides in ("both", "upper") and self.bc_type != "periodic":
            u_high = u[_make_slice(rank, axis, slice(-1, None))]
            if self.bc_type == "dirichlet":
                r_high = ops.subtract(u_high, g)
            elif self.bc_type == "neumann":
                # Outward normal is +e_axis: du/dn = (u_last - u_prev) / h
                u_prev = u[_make_slice(rank, axis, slice(-2, -1))]
                du_dn_high = ops.divide(ops.subtract(u_high, u_prev), h)
                r_high = ops.subtract(du_dn_high, g)
            elif self.bc_type == "robin":
                u_prev = u[_make_slice(rank, axis, slice(-2, -1))]
                du_dn_high = ops.divide(ops.subtract(u_high, u_prev), h)
                robin_term = ops.add(
                    ops.cast(self.alpha, self.compute_dtype) * u_high,
                    ops.cast(self.beta, self.compute_dtype) * du_dn_high,
                )
                r_high = ops.subtract(robin_term, g)
            residuals.append(r_high)

        return residuals

    def _apply_hard_dirichlet(self, u: Any, axis: int) -> Any:
        rank = len(u.shape)
        u_slice_0 = u[_make_slice(rank, axis, slice(0, 1))]
        u_slice_end = u[_make_slice(rank, axis, slice(-1, None))]

        g_val = ops.cast(self.value, self.compute_dtype)
        g_low = ops.add(ops.zeros_like(u_slice_0), g_val)
        g_high = ops.add(ops.zeros_like(u_slice_end), g_val)

        if self.sides == "both":
            interior = u[_make_slice(rank, axis, slice(1, -1))]
            return ops.concatenate([g_low, interior, g_high], axis=axis)
        elif self.sides == "lower":
            rest = u[_make_slice(rank, axis, slice(1, None))]
            return ops.concatenate([g_low, rest], axis=axis)
        elif self.sides == "upper":
            rest = u[_make_slice(rank, axis, slice(0, -1))]
            return ops.concatenate([rest, g_high], axis=axis)
        return u


    def compute_output_shape(self, input_shape: Any) -> Any:
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            # Collocation tuple input
            u_shape = input_shape[0]
            if self.return_residual:
                return u_shape, u_shape
            return u_shape

        if self.return_residual:
            return input_shape, (None,)
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "bc_type": self.bc_type,
                "value": self.value,
                "axes": self.axes,
                "sides": self.sides,
                "dx": self.dx,
                "alpha": self.alpha,
                "beta": self.beta,
                "weight": self.weight,
                "add_loss": self.add_loss_enabled,
                "enforcement": self.enforcement,
                "return_residual": self.return_residual,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BoundaryCondition":
        return cls(**config)


@keras.saving.register_keras_serializable(package="keras_sciml")
class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition ($u = g$) convenience layer."""

    def __init__(self, value: Union[float, Any] = 0.0, **kwargs: Any):
        kwargs.pop("bc_type", None)
        super().__init__(bc_type="dirichlet", value=value, **kwargs)


@keras.saving.register_keras_serializable(package="keras_sciml")
class NeumannBC(BoundaryCondition):
    """Neumann boundary condition ($\\frac{\\partial u}{\\partial n} = g$) convenience layer."""

    def __init__(self, value: Union[float, Any] = 0.0, **kwargs: Any):
        kwargs.pop("bc_type", None)
        super().__init__(bc_type="neumann", value=value, **kwargs)


@keras.saving.register_keras_serializable(package="keras_sciml")
class RobinBC(BoundaryCondition):
    """Robin boundary condition ($\\alpha u + \\beta \\frac{\\partial u}{\\partial n} = g$) convenience layer."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, value: Union[float, Any] = 0.0, **kwargs: Any):
        kwargs.pop("bc_type", None)
        super().__init__(bc_type="robin", alpha=alpha, beta=beta, value=value, **kwargs)


@keras.saving.register_keras_serializable(package="keras_sciml")
class PeriodicBC(BoundaryCondition):
    """Periodic boundary condition ($u|_{\\text{lower}} = u|_{\\text{upper}}$) convenience layer."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("bc_type", None)
        super().__init__(bc_type="periodic", **kwargs)

