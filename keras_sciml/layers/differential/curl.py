from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.layers import Layer

from keras_sciml.layers.differential.gradient import _differentiate_axis


@keras.saving.register_keras_serializable(package="keras_sciml")
class Curl(Layer):
    """Spatial Curl (nabla times v) layer using finite differences.

    Computes the curl (vorticity) of a vector field across 2D or 3D spatial grids:
    - 2D scalar curl (vorticity):
      $$\\text{curl}_{2D}(\\mathbf{v}) = \\frac{\\partial v_y}{\\partial x} - \\frac{\\partial v_x}{\\partial y}$$
      For input shape `(batch, x, y, 2)`, produces output shape `(batch, x, y, 1)` (or `(batch, x, y)` if `keepdims=False`).
    - 3D vector curl:
      $$\\nabla \\times \\mathbf{v} = \\begin{bmatrix}
      \\frac{\\partial v_z}{\\partial y} - \\frac{\\partial v_y}{\\partial z} \\\\
      \\frac{\\partial v_x}{\\partial z} - \\frac{\\partial v_z}{\\partial x} \\\\
      \\frac{\\partial v_y}{\\partial x} - \\frac{\\partial v_x}{\\partial y}
      \\end{bmatrix}$$
      For input shape `(batch, x, y, z, 3)`, produces output shape `(batch, x, y, z, 3)`.

    Also supports multiple vector fields packed into the channels dimension:
    If the channel dimension is `M * d` (where `d` is 2 or 3), computes the curl for each of
    the `M` vector fields:
    - In 2D: Produces `M` scalar curl components.
    - In 3D: Produces `3 * M` vector curl components.

    Args:
        dx: float or sequence of floats, grid spacing along each spatial dimension.
            Defaults to 1.0.
        boundary_condition: str, boundary treatment for finite differences.
            Supported options:
            - `"central"`: Second-order central differences in interior, one-sided differences
              at boundaries (matches `numpy.gradient`).
            - `"periodic"`: Periodic wrap-around boundary conditions using cyclic rolls.
            Defaults to `"central"`.
        keepdims: bool, whether to keep the trailing dimension of size `M` in 2D (defaults to True).
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        dx: Union[float, Sequence[float]] = 1.0,
        boundary_condition: str = "central",
        keepdims: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if boundary_condition not in ("central", "periodic"):
            raise ValueError(
                f"`boundary_condition` must be 'central' or 'periodic', received: {boundary_condition}"
            )

        self.dx = dx
        self.boundary_condition = boundary_condition
        self.keepdims = bool(keepdims)

    def _get_spacings(self, spatial_dim_count: int) -> List[float]:
        if isinstance(self.dx, (int, float)):
            return [float(self.dx)] * spatial_dim_count
        elif isinstance(self.dx, (list, tuple)):
            if len(self.dx) != spatial_dim_count:
                raise ValueError(
                    f"Length of `dx` ({len(self.dx)}) must match number of spatial dimensions ({spatial_dim_count})."
                )
            return [float(s) for s in self.dx]
        else:
            raise TypeError(f"Unsupported type for `dx`: {type(self.dx)}")

    def call(self, inputs: Any) -> Any:
        v = ops.cast(inputs, self.compute_dtype)
        rank = len(v.shape)
        spatial_dim_count = rank - 2
        if spatial_dim_count not in (2, 3):
            raise ValueError(
                f"`Curl` is only defined for 2D and 3D spatial grids. "
                f"Received tensor with rank {rank} ({spatial_dim_count} spatial dimensions)."
            )

        channels = v.shape[-1]
        if channels is not None and channels % spatial_dim_count != 0:
            raise ValueError(
                f"The channel dimension ({channels}) of inputs to `Curl` must be divisible by "
                f"the number of spatial dimensions ({spatial_dim_count})."
            )

        spacings = self._get_spacings(spatial_dim_count)
        num_fields = (channels // spatial_dim_count) if channels is not None else 1

        field_curls = []
        if spatial_dim_count == 2:
            # 2D Curl: dvy/dx - dvx/dy
            # axis 1 is x (h = spacings[0])
            # axis 2 is y (h = spacings[1])
            for m in range(num_fields):
                vx = v[..., m * 2 : m * 2 + 1]
                vy = v[..., m * 2 + 1 : m * 2 + 2]

                dvy_dx = _differentiate_axis(
                    vy,
                    axis=1,
                    h=spacings[0],
                    boundary_condition=self.boundary_condition,
                )
                dvx_dy = _differentiate_axis(
                    vx,
                    axis=2,
                    h=spacings[1],
                    boundary_condition=self.boundary_condition,
                )
                curl_m = ops.subtract(dvy_dx, dvx_dy)
                field_curls.append(curl_m)

            if len(field_curls) == 1:
                out = field_curls[0]
            else:
                out = ops.concatenate(field_curls, axis=-1)

            if not self.keepdims and out.shape[-1] == 1:
                out = ops.squeeze(out, axis=-1)

            return out

        else:
            # 3D Curl:
            # curl_x = dvz/dy - dvy/dz
            # curl_y = dvx/dz - dvz/dx
            # curl_z = dvy/dx - dvx/dy
            # axis 1 is x (h = spacings[0])
            # axis 2 is y (h = spacings[1])
            # axis 3 is z (h = spacings[2])
            for m in range(num_fields):
                vx = v[..., m * 3 : m * 3 + 1]
                vy = v[..., m * 3 + 1 : m * 3 + 2]
                vz = v[..., m * 3 + 2 : m * 3 + 3]

                # dvz/dy (axis 2, h 1) and dvy/dz (axis 3, h 2)
                dvz_dy = _differentiate_axis(
                    vz, axis=2, h=spacings[1], boundary_condition=self.boundary_condition
                )
                dvy_dz = _differentiate_axis(
                    vy, axis=3, h=spacings[2], boundary_condition=self.boundary_condition
                )
                curl_x = ops.subtract(dvz_dy, dvy_dz)

                # dvx/dz (axis 3, h 2) and dvz/dx (axis 1, h 0)
                dvx_dz = _differentiate_axis(
                    vx, axis=3, h=spacings[2], boundary_condition=self.boundary_condition
                )
                dvz_dx = _differentiate_axis(
                    vz, axis=1, h=spacings[0], boundary_condition=self.boundary_condition
                )
                curl_y = ops.subtract(dvx_dz, dvz_dx)

                # dvy/dx (axis 1, h 0) and dvx/dy (axis 2, h 1)
                dvy_dx = _differentiate_axis(
                    vy, axis=1, h=spacings[0], boundary_condition=self.boundary_condition
                )
                dvx_dy = _differentiate_axis(
                    vx, axis=2, h=spacings[1], boundary_condition=self.boundary_condition
                )
                curl_z = ops.subtract(dvy_dx, dvx_dy)

                curl_m = ops.concatenate([curl_x, curl_y, curl_z], axis=-1)
                field_curls.append(curl_m)

            if len(field_curls) == 1:
                return field_curls[0]
            else:
                return ops.concatenate(field_curls, axis=-1)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        rank = len(input_shape)
        spatial_dim_count = rank - 2
        in_shape = list(input_shape)
        channels = in_shape[-1]
        num_fields = (channels // spatial_dim_count) if channels is not None else 1

        if spatial_dim_count == 2:
            if not self.keepdims and num_fields == 1:
                return tuple(in_shape[:-1])
            else:
                return tuple(in_shape[:-1] + [num_fields])
        elif spatial_dim_count == 3:
            return tuple(in_shape[:-1] + [num_fields * 3])
        else:
            return tuple(in_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dx": self.dx,
                "boundary_condition": self.boundary_condition,
                "keepdims": self.keepdims,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Curl":
        return cls(**config)

