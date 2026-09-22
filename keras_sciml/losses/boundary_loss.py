from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.losses import Loss


def _make_slice(rank: int, axis: int, sl: slice) -> Tuple[slice, ...]:
    """Helper to create a slicing tuple for a specific axis."""
    slices = [slice(None)] * rank
    slices[axis] = sl
    return tuple(slices)


@keras.saving.register_keras_serializable(package="keras_sciml")
class BoundaryLoss(Loss):
    """Boundary condition loss for physics-informed learning and neural operators.

    Computes discrepancy penalties on spatial boundaries for:
    - `"dirichlet"`: $u|_{\\partial\\Omega} - g$
    - `"neumann"`: $\\frac{\\partial u}{\\partial n}|_{\\partial\\Omega} - g$
    - `"robin"`: $\\alpha u|_{\\partial\\Omega} + \\beta \\frac{\\partial u}{\\partial n}|_{\\partial\\Omega} - g$
    - `"periodic"`: $u|_{\\text{lower}} - u|_{\\text{upper}}$

    Supports both spatial grid predictions (where boundary faces are extracted along `axes`)
    and explicit boundary collocation arrays.

    Args:
        bc_type: str, type of boundary condition (`"dirichlet"`, `"neumann"`, `"robin"`, `"periodic"`).
            Defaults to `"dirichlet"`.
        value: float or None, default prescribed target value on boundary. If None and `y_true` is provided,
            boundary targets are extracted from `y_true`. Defaults to None.
        axes: int or sequence of ints, spatial axes along which boundaries are evaluated.
            If None, inferred as all spatial dimensions (`range(1, rank - 1)` for channel-last tensors).
        sides: str, boundary faces to evaluate (`"both"`, `"lower"`, `"upper"`). Defaults to `"both"`.
        dx: float or sequence of floats, grid spacing along each spatial axis (for normal derivatives).
            Defaults to 1.0.
        alpha: float, coefficient for Robin boundary value. Defaults to 1.0.
        beta: float, coefficient for Robin normal derivative. Defaults to 1.0.
        loss_type: str, loss penalty norm (`"mse"` or `"mae"`). Defaults to `"mse"`.
        name: str, loss name. Defaults to `"boundary_loss"`.
        reduction: str, Keras loss reduction mode. Defaults to `"sum_over_batch_size"`.
    """

    def __init__(
        self,
        bc_type: str = "dirichlet",
        value: Optional[float] = None,
        axes: Optional[Union[int, Sequence[int]]] = None,
        sides: str = "both",
        dx: Union[float, Sequence[float]] = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        loss_type: str = "mse",
        name: str = "boundary_loss",
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__(name=name, reduction=reduction)
        bc_type = bc_type.lower()
        if bc_type not in ("dirichlet", "neumann", "robin", "periodic"):
            raise ValueError(
                f"`bc_type` must be one of ('dirichlet', 'neumann', 'robin', 'periodic'), received: {bc_type}"
            )

        sides = sides.lower()
        if sides not in ("both", "lower", "upper"):
            raise ValueError(f"`sides` must be one of ('both', 'lower', 'upper'), received: {sides}")

        loss_type = loss_type.lower()
        if loss_type not in ("mse", "mae"):
            raise ValueError(f"`loss_type` must be one of ('mse', 'mae'), received: {loss_type}")

        self.bc_type = bc_type
        self.value = float(value) if value is not None else None
        if axes is not None:
            self.axes = (axes,) if isinstance(axes, int) else tuple(axes)
        else:
            self.axes = None
        self.sides = sides
        self.dx = dx
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.loss_type = loss_type

    def _get_target_axes(self, rank: int) -> List[int]:
        if self.axes is not None:
            return [(ax if ax >= 0 else rank + ax) for ax in self.axes]
        if rank >= 3:
            return list(range(1, rank - 1))
        elif rank == 2:
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
                return {ax: float(self.dx[ax - 1]) for ax in target_axes}
            else:
                raise ValueError(
                    f"Length of `dx` ({len(self.dx)}) must match number of target axes ({len(target_axes)}) "
                    f"or total spatial dimensions ({spatial_dim_count})."
                )
        else:
            raise TypeError(f"Unsupported type for `dx`: {type(self.dx)}")

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_pred = ops.convert_to_tensor(y_pred)
        if y_true is not None:
            y_true = ops.cast(y_true, dtype=y_pred.dtype)

        rank = ops.ndim(y_pred)

        # Collocation / point-cloud prediction (e.g. rank 1 or shape matches boundary target directly)
        if rank <= 1:
            if self.value is not None:
                target = ops.cast(self.value, y_pred.dtype)
            elif y_true is not None:
                target = y_true
            else:
                target = ops.zeros_like(y_pred)
            diff = y_pred - target
            return ops.square(diff) if self.loss_type == "mse" else ops.abs(diff)

        target_axes = self._get_target_axes(rank)
        spacings = self._get_spacings(target_axes, rank)

        same_shape = y_true is not None and (
            hasattr(y_true, "shape") and hasattr(y_pred, "shape") and y_true.shape == y_pred.shape
        )

        face_losses = []
        for axis in target_axes:
            h = spacings[axis]

            # Lower boundary: index 0
            if self.sides in ("both", "lower"):
                u_low = y_pred[_make_slice(rank, axis, slice(0, 1))]
                if self.bc_type == "dirichlet":
                    if self.value is not None:
                        g_low = ops.cast(self.value, u_low.dtype)
                    elif same_shape:
                        g_low = y_true[_make_slice(rank, axis, slice(0, 1))]
                    else:
                        g_low = ops.zeros_like(u_low)
                    r_low = u_low - g_low
                elif self.bc_type == "neumann":
                    # Outward normal is -e_axis: du/dn = (u_0 - u_1) / h
                    u_next = y_pred[_make_slice(rank, axis, slice(1, 2))]
                    du_dn_low = (u_low - u_next) / h
                    if self.value is not None:
                        g_low = ops.cast(self.value, du_dn_low.dtype)
                    elif same_shape:
                        yt_low = y_true[_make_slice(rank, axis, slice(0, 1))]
                        yt_next = y_true[_make_slice(rank, axis, slice(1, 2))]
                        g_low = (yt_low - yt_next) / h
                    else:
                        g_low = ops.zeros_like(du_dn_low)
                    r_low = du_dn_low - g_low
                elif self.bc_type == "robin":
                    u_next = y_pred[_make_slice(rank, axis, slice(1, 2))]
                    du_dn_low = (u_low - u_next) / h
                    robin_pred = self.alpha * u_low + self.beta * du_dn_low
                    if self.value is not None:
                        g_low = ops.cast(self.value, robin_pred.dtype)
                    elif same_shape:
                        yt_low = y_true[_make_slice(rank, axis, slice(0, 1))]
                        yt_next = y_true[_make_slice(rank, axis, slice(1, 2))]
                        g_low = self.alpha * yt_low + self.beta * ((yt_low - yt_next) / h)
                    else:
                        g_low = ops.zeros_like(robin_pred)
                    r_low = robin_pred - g_low
                elif self.bc_type == "periodic":
                    u_high = y_pred[_make_slice(rank, axis, slice(-1, None))]
                    r_low = u_low - u_high

                reduce_axes = tuple(range(1, rank))
                loss_low = (
                    ops.mean(ops.square(r_low), axis=reduce_axes)
                    if self.loss_type == "mse"
                    else ops.mean(ops.abs(r_low), axis=reduce_axes)
                )
                face_losses.append(loss_low)

            # Upper boundary: index -1
            if self.sides in ("both", "upper") and self.bc_type != "periodic":
                u_high = y_pred[_make_slice(rank, axis, slice(-1, None))]
                if self.bc_type == "dirichlet":
                    if self.value is not None:
                        g_high = ops.cast(self.value, u_high.dtype)
                    elif same_shape:
                        g_high = y_true[_make_slice(rank, axis, slice(-1, None))]
                    else:
                        g_high = ops.zeros_like(u_high)
                    r_high = u_high - g_high
                elif self.bc_type == "neumann":
                    # Outward normal is +e_axis: du/dn = (u_last - u_prev) / h
                    u_prev = y_pred[_make_slice(rank, axis, slice(-2, -1))]
                    du_dn_high = (u_high - u_prev) / h
                    if self.value is not None:
                        g_high = ops.cast(self.value, du_dn_high.dtype)
                    elif same_shape:
                        yt_high = y_true[_make_slice(rank, axis, slice(-1, None))]
                        yt_prev = y_true[_make_slice(rank, axis, slice(-2, -1))]
                        g_high = (yt_high - yt_prev) / h
                    else:
                        g_high = ops.zeros_like(du_dn_high)
                    r_high = du_dn_high - g_high
                elif self.bc_type == "robin":
                    u_prev = y_pred[_make_slice(rank, axis, slice(-2, -1))]
                    du_dn_high = (u_high - u_prev) / h
                    robin_pred = self.alpha * u_high + self.beta * du_dn_high
                    if self.value is not None:
                        g_high = ops.cast(self.value, robin_pred.dtype)
                    elif same_shape:
                        yt_high = y_true[_make_slice(rank, axis, slice(-1, None))]
                        yt_prev = y_true[_make_slice(rank, axis, slice(-2, -1))]
                        g_high = self.alpha * yt_high + self.beta * ((yt_high - yt_prev) / h)
                    else:
                        g_high = ops.zeros_like(robin_pred)
                    r_high = robin_pred - g_high

                reduce_axes = tuple(range(1, rank))
                loss_high = (
                    ops.mean(ops.square(r_high), axis=reduce_axes)
                    if self.loss_type == "mse"
                    else ops.mean(ops.abs(r_high), axis=reduce_axes)
                )
                face_losses.append(loss_high)

        if not face_losses:
            return ops.zeros((ops.shape(y_pred)[0],), dtype=y_pred.dtype)

        total_loss = face_losses[0]
        for f_loss in face_losses[1:]:
            total_loss = total_loss + f_loss
        return total_loss

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "bc_type": self.bc_type,
            "value": self.value,
            "axes": self.axes,
            "sides": self.sides,
            "dx": self.dx,
            "alpha": self.alpha,
            "beta": self.beta,
            "loss_type": self.loss_type,
        }
