from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.layers import Layer

from keras_sciml.layers.differential.gradient import _differentiate_axis


@keras.saving.register_keras_serializable(package="keras_sciml")
class Divergence(Layer):
    """Spatial Divergence (nabla cdot v) layer using finite differences.

    Computes the divergence of a vector field across spatial grid dimensions:
        div(v) = sum_{i=1}^d frac{partial v_i}{partial x_i}

    Supports 1D, 2D, and 3D spatial grids in channels-last format:
    - 1D: `(batch, x, 1)` -> output `(batch, x, 1)`
    - 2D: `(batch, x, y, 2)` -> output `(batch, x, y, 1)`
    - 3D: `(batch, x, y, z, 3)` -> output `(batch, x, y, z, 1)`

    Also supports multiple vector fields packed into the channels dimension:
    If the channel dimension is `M * d`, computes the divergence for each of the `M` vector
    fields, producing an output with `M` channels.

    Args:
        dx: float or sequence of floats, grid spacing along each spatial dimension.
            Defaults to 1.0.
        boundary_condition: str, boundary treatment for finite differences.
            Supported options:
            - `"central"`: Second-order central differences in interior, one-sided differences
              at boundaries (matches `numpy.gradient`).
            - `"periodic"`: Periodic wrap-around boundary conditions using cyclic rolls.
            Defaults to `"central"`.
        keepdims: bool, whether to keep the trailing dimension of size `M` (defaults to True).
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
        if spatial_dim_count < 1:
            raise ValueError(
                f"`Divergence` expects inputs with at least 1 spatial dimension and a component dimension. "
                f"Received rank: {rank}"
            )

        channels = v.shape[-1]
        if channels is not None and channels % spatial_dim_count != 0:
            raise ValueError(
                f"The channel dimension ({channels}) of inputs to `Divergence` must be divisible by "
                f"the number of spatial dimensions ({spatial_dim_count})."
            )

        spacings = self._get_spacings(spatial_dim_count)
        num_fields = (channels // spatial_dim_count) if channels is not None else 1

        field_divergences = []
        for m in range(num_fields):
            div_terms = []
            for i in range(spatial_dim_count):
                comp_idx = m * spatial_dim_count + i
                spatial_axis = i + 1
                v_i = v[..., comp_idx : comp_idx + 1]
                h_i = spacings[i]
                dvi_dxi = _differentiate_axis(
                    v_i,
                    axis=spatial_axis,
                    h=h_i,
                    boundary_condition=self.boundary_condition,
                )
                div_terms.append(dvi_dxi)

            field_div = div_terms[0]
            for term in div_terms[1:]:
                field_div = ops.add(field_div, term)
            field_divergences.append(field_div)

        if len(field_divergences) == 1:
            out = field_divergences[0]
        else:
            out = ops.concatenate(field_divergences, axis=-1)

        if not self.keepdims and out.shape[-1] == 1:
            out = ops.squeeze(out, axis=-1)

        return out

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        rank = len(input_shape)
        spatial_dim_count = rank - 2
        in_shape = list(input_shape)
        channels = in_shape[-1]
        num_fields = (channels // spatial_dim_count) if channels is not None else 1

        if not self.keepdims and num_fields == 1:
            return tuple(in_shape[:-1])
        else:
            return tuple(in_shape[:-1] + [num_fields])

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
    def from_config(cls, config: Dict[str, Any]) -> "Divergence":
        return cls(**config)

