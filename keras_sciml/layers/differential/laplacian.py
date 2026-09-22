from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.layers import Layer


def _second_derivative_axis(
    u: Any,
    axis: int,
    h: float,
    boundary_condition: str = "central",
) -> Any:
    """Computes second-order accurate second derivative d^2u / dx^2 along an axis."""
    h_sq = ops.cast(h * h, u.dtype)

    if boundary_condition == "periodic":
        forward_shift = ops.roll(u, shift=-1, axis=axis)
        backward_shift = ops.roll(u, shift=1, axis=axis)
        # (u_{i+1} - 2*u_i + u_{i-1}) / h^2
        numerator = ops.subtract(
            ops.add(forward_shift, backward_shift),
            ops.multiply(u, 2.0),
        )
        return ops.divide(numerator, h_sq)

    rank = len(u.shape)

    def _make_slice(start: Optional[int], stop: Optional[int]) -> Tuple[Union[slice, int], ...]:
        return tuple(slice(start, stop) if i == axis else slice(None) for i in range(rank))

    # Interior: (u_{i+1} - 2*u_i + u_{i-1}) / h^2
    mid_num = ops.subtract(
        ops.add(u[_make_slice(2, None)], u[_make_slice(None, -2)]),
        ops.multiply(u[_make_slice(1, -1)], 2.0),
    )
    mid = ops.divide(mid_num, h_sq)

    # Left boundary (second-order forward): (2*u_0 - 5*u_1 + 4*u_2 - u_3) / h^2
    left_num = ops.subtract(
        ops.add(
            ops.multiply(u[_make_slice(0, 1)], 2.0),
            ops.multiply(u[_make_slice(2, 3)], 4.0),
        ),
        ops.add(
            ops.multiply(u[_make_slice(1, 2)], 5.0),
            u[_make_slice(3, 4)],
        ),
    )
    left = ops.divide(left_num, h_sq)

    # Right boundary (second-order backward): (2*u_{N-1} - 5*u_{N-2} + 4*u_{N-3} - u_{N-4}) / h^2
    right_num = ops.subtract(
        ops.add(
            ops.multiply(u[_make_slice(-1, None)], 2.0),
            ops.multiply(u[_make_slice(-3, -2)], 4.0),
        ),
        ops.add(
            ops.multiply(u[_make_slice(-2, -1)], 5.0),
            u[_make_slice(-4, -3)],
        ),
    )
    right = ops.divide(right_num, h_sq)

    return ops.concatenate([left, mid, right], axis=axis)


@keras.saving.register_keras_serializable(package="keras_sciml")
class Laplacian(Layer):
    """Spatial Laplacian (nabla^2 u) layer using finite differences.

    Computes the scalar Laplacian of a field across spatial dimensions:
        Delta u = sum_{i=1}^d frac{partial^2 u}{partial x_i^2}

    Supports 1D, 2D, and 3D spatial grids in channels-last format:
    - 1D: `(batch, x, channels)`
    - 2D: `(batch, x, y, channels)`
    - 3D: `(batch, x, y, z, channels)`

    Args:
        dx: float or sequence of floats, grid spacing along each spatial dimension.
            Defaults to 1.0.
        boundary_condition: str, boundary treatment for finite differences.
            Supported options:
            - `"central"`: Second-order central differences in interior with second-order
              one-sided formulas at boundaries.
            - `"periodic"`: Periodic wrap-around boundary conditions using cyclic rolls.
            Defaults to `"central"`.
        axes: Optional integer or sequence of integers specifying which spatial axes
            to compute second derivatives along. If None, computes along all spatial axes `1 ... rank-2`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        dx: Union[float, Sequence[float]] = 1.0,
        boundary_condition: str = "central",
        axes: Optional[Union[int, Sequence[int]]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if boundary_condition not in ("central", "periodic"):
            raise ValueError(
                f"`boundary_condition` must be 'central' or 'periodic', received: {boundary_condition}"
            )

        self.dx = dx
        self.boundary_condition = boundary_condition
        self.axes = [axes] if isinstance(axes, int) else (list(axes) if axes is not None else None)

    def _get_target_axes_and_spacings(self, rank: int) -> Tuple[List[int], List[float]]:
        spatial_dim_count = rank - 2
        if spatial_dim_count < 1:
            raise ValueError(
                f"`Laplacian` expects inputs with at least 1 spatial dimension and a channel dimension "
                f"(rank >= 3). Received rank: {rank}"
            )

        default_axes = list(range(1, rank - 1))
        target_axes = self.axes if self.axes is not None else default_axes
        target_axes = [ax % rank for ax in target_axes]

        if isinstance(self.dx, (int, float)):
            spacings = [float(self.dx)] * len(target_axes)
        elif isinstance(self.dx, (list, tuple)):
            if len(self.dx) == len(target_axes):
                spacings = [float(s) for s in self.dx]
            elif len(self.dx) == spatial_dim_count:
                spacings = [float(self.dx[ax - 1]) for ax in target_axes]
            else:
                raise ValueError(
                    f"Length of `dx` ({len(self.dx)}) must match number of target axes ({len(target_axes)}) "
                    f"or total spatial dimensions ({spatial_dim_count})."
                )
        else:
            raise TypeError(f"Unsupported type for `dx`: {type(self.dx)}")

        return target_axes, spacings

    def call(self, inputs: Any) -> Any:
        u = ops.cast(inputs, self.compute_dtype)
        rank = len(u.shape)
        target_axes, spacings = self._get_target_axes_and_spacings(rank)

        laplacian_terms = []
        for axis, h in zip(target_axes, spacings):
            d2u = _second_derivative_axis(
                u, axis=axis, h=h, boundary_condition=self.boundary_condition
            )
            laplacian_terms.append(d2u)

        out = laplacian_terms[0]
        for term in laplacian_terms[1:]:
            out = ops.add(out, term)

        return out

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dx": self.dx,
                "boundary_condition": self.boundary_condition,
                "axes": self.axes,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Laplacian":
        return cls(**config)

