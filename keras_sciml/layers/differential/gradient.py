from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.layers import Layer


def _differentiate_axis(
    u: Any,
    axis: int,
    h: float,
    boundary_condition: str = "central",
) -> Any:
    """Computes second-order finite differences along a given axis."""
    h_const = ops.cast(h, u.dtype)

    if boundary_condition == "periodic":
        forward_shift = ops.roll(u, shift=-1, axis=axis)
        backward_shift = ops.roll(u, shift=1, axis=axis)
        return ops.divide(ops.subtract(forward_shift, backward_shift), 2.0 * h_const)

    rank = len(u.shape)

    def _make_slice(start: Optional[int], stop: Optional[int]) -> Tuple[Union[slice, int], ...]:
        return tuple(slice(start, stop) if i == axis else slice(None) for i in range(rank))

    left = ops.divide(
        ops.subtract(u[_make_slice(1, 2)], u[_make_slice(0, 1)]),
        h_const,
    )
    interior = ops.divide(
        ops.subtract(u[_make_slice(2, None)], u[_make_slice(None, -2)]),
        2.0 * h_const,
    )
    right = ops.divide(
        ops.subtract(u[_make_slice(-1, None)], u[_make_slice(-2, -1)]),
        h_const,
    )
    return ops.concatenate([left, interior, right], axis=axis)


@keras.saving.register_keras_serializable(package="keras_sciml")
class Gradient(Layer):
    """Spatial Gradient layer using finite differences.

    Computes directional derivatives along specified spatial grid axes:
        grad(u) = [du/dx_1, du/dx_2, ..., du/dx_d]

    Supports 1D, 2D, and 3D spatial grids in channels-last format:
    - 1D: `(batch, x, channels)`
    - 2D: `(batch, x, y, channels)`
    - 3D: `(batch, x, y, z, channels)`

    Args:
        dx: float or sequence of floats, grid spacing along each spatial dimension.
            Defaults to 1.0.
        boundary_condition: str, boundary treatment for finite differences.
            Supported options:
            - `"central"`: Second-order central differences in interior, one-sided differences
              at boundaries (matches `numpy.gradient`).
            - `"periodic"`: Periodic wrap-around boundary conditions using cyclic rolls.
            Defaults to `"central"`.
        axes: Optional integer or sequence of integers specifying which spatial axes
            to differentiate along. If None, differentiates along all spatial axes `1 ... rank-2`.
        concat_components: bool, whether to concatenate directional derivatives along
            the channel axis (producing `shape=(..., channels * d)`) or add a new trailing
            component axis (producing `shape=(..., channels, d)`). Defaults to False.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        dx: Union[float, Sequence[float]] = 1.0,
        boundary_condition: str = "central",
        axes: Optional[Union[int, Sequence[int]]] = None,
        concat_components: bool = False,
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
        self.concat_components = bool(concat_components)

    def _get_target_axes_and_spacings(self, rank: int) -> Tuple[List[int], List[float]]:
        spatial_dim_count = rank - 2
        if spatial_dim_count < 1:
            raise ValueError(
                f"`Gradient` expects inputs with at least 1 spatial dimension and a channel dimension "
                f"(rank >= 3). Received rank: {rank}"
            )

        default_axes = list(range(1, rank - 1))
        target_axes = self.axes if self.axes is not None else default_axes

        # Normalize negative axis indices
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

        grad_components = []
        for axis, h in zip(target_axes, spacings):
            du = _differentiate_axis(u, axis=axis, h=h, boundary_condition=self.boundary_condition)
            grad_components.append(du)

        if self.concat_components:
            return ops.concatenate(grad_components, axis=-1)
        else:
            return ops.stack(grad_components, axis=-1)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        rank = len(input_shape)
        target_axes, _ = self._get_target_axes_and_spacings(rank)
        num_components = len(target_axes)
        in_shape = list(input_shape)
        channels = in_shape[-1]

        if self.concat_components:
            out_channels = channels * num_components if channels is not None else None
            return tuple(in_shape[:-1] + [out_channels])
        else:
            return tuple(in_shape + [num_components])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dx": self.dx,
                "boundary_condition": self.boundary_condition,
                "axes": self.axes,
                "concat_components": self.concat_components,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Gradient":
        return cls(**config)

