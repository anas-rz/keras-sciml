from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.layers import Layer

from keras_sciml.layers.differential.gradient import Gradient
from keras_sciml.layers.differential.laplacian import Laplacian


@keras.saving.register_keras_serializable(package="keras_sciml")
class Residual(Layer):
    """Physics residual evaluation and loss penalty layer for PINNs and neural operators.

    Evaluates the PDE residual $\\mathcal{R}(u) = \\mathcal{N}(u) - f$ and attaches the
    resulting physics penalty loss directly to the layer via `self.add_loss`.

    Supported canonical equations (`equation`):
    - `"poisson"`: $-\\Delta u - f = 0$
    - `"diffusion"`: $-\\alpha \\Delta u - f = 0$
    - `"burgers"`: $u \\frac{\\partial u}{\\partial x} - \\nu \\Delta u - f = 0$
    - `"advection"`: $c \\frac{\\partial u}{\\partial x} - f = 0$
    - `"custom"`: Arbitrary user-defined residual function via `pde_fn`.

    Inputs:
    - Single tensor `u`: uses static `forcing` specified at layer initialization.
    - Tuple `(u, f)`: allows dynamic per-sample forcing terms $f$ (crucial for PINO & operator learning).

    Args:
        equation: str, name of the PDE equation (`"poisson"`, `"diffusion"`, `"burgers"`, `"advection"`, `"custom"`).
            Defaults to `"poisson"`.
        pde_fn: Optional callable for `"custom"` equations: `res = pde_fn(u)` or `res = pde_fn(u, f)`.
        dx: float or sequence of floats, spatial grid spacing. Defaults to 1.0.
        diffusivity: float, diffusion coefficient $\\alpha$ for `"diffusion"`. Defaults to 1.0.
        nu: float, kinematic viscosity $\\nu$ for `"burgers"`. Defaults to 0.01.
        velocity: float, advection speed $c$ for `"advection"`. Defaults to 1.0.
        forcing: float or tensor, static source/forcing term $f$. Defaults to 0.0.
        boundary_condition: str, boundary treatment for finite differences (`"central"` or `"periodic"`).
            Defaults to `"central"`.
        weight: float, loss weighting factor when `add_loss=True`. Defaults to 1.0.
        add_loss: bool, whether to register the physics MSE loss with `self.add_loss`. Defaults to True.
        return_residual: bool, whether to return `(output, residual)` instead of just `output`.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        equation: str = "poisson",
        pde_fn: Optional[Callable[..., Any]] = None,
        dx: Union[float, Sequence[float]] = 1.0,
        diffusivity: float = 1.0,
        nu: float = 0.01,
        velocity: float = 1.0,
        forcing: Union[float, Any] = 0.0,
        boundary_condition: str = "central",
        weight: float = 1.0,
        add_loss: bool = True,
        return_residual: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        equation = equation.lower()
        if equation not in ("poisson", "diffusion", "burgers", "advection", "custom"):
            raise ValueError(
                f"`equation` must be one of ('poisson', 'diffusion', 'burgers', 'advection', 'custom'), "
                f"received: {equation}"
            )

        if equation == "custom" and pde_fn is None:
            raise ValueError("When `equation='custom'`, a callable `pde_fn` must be provided.")

        self.equation = equation
        self.pde_fn = pde_fn
        self.dx = dx
        self.diffusivity = float(diffusivity)
        self.nu = float(nu)
        self.velocity = float(velocity)
        self.forcing = forcing
        self.boundary_condition = boundary_condition
        self.weight = float(weight)
        self.add_loss_enabled = bool(add_loss)
        self.return_residual = bool(return_residual)

        # Build differential operators
        if self.equation in ("poisson", "diffusion", "burgers"):
            self.laplacian = Laplacian(dx=self.dx, boundary_condition=self.boundary_condition)
        else:
            self.laplacian = None

        if self.equation in ("burgers", "advection"):
            self.gradient = Gradient(
                dx=self.dx,
                boundary_condition=self.boundary_condition,
                axes=[1],
                concat_components=True,
            )
        else:
            self.gradient = None

    def call(self, inputs: Any) -> Any:
        if isinstance(inputs, (tuple, list)):
            u, f = inputs
            u = ops.cast(u, self.compute_dtype)
            f = ops.cast(f, self.compute_dtype)
        else:
            u = ops.cast(inputs, self.compute_dtype)
            f = ops.cast(self.forcing, self.compute_dtype)

        if self.equation == "poisson":
            # -Delta u - f = 0
            lap = self.laplacian(u)
            residual = ops.subtract(ops.negative(lap), f)
        elif self.equation == "diffusion":
            # -alpha * Delta u - f = 0
            lap = self.laplacian(u)
            scaled_lap = ops.multiply(ops.cast(self.diffusivity, self.compute_dtype), lap)
            residual = ops.subtract(ops.negative(scaled_lap), f)
        elif self.equation == "burgers":
            # u * du/dx - nu * Delta u - f = 0
            du_dx = self.gradient(u)
            adv = ops.multiply(u, du_dx)
            diff = ops.multiply(ops.cast(self.nu, self.compute_dtype), self.laplacian(u))
            residual = ops.subtract(ops.subtract(adv, diff), f)
        elif self.equation == "advection":
            # c * du/dx - f = 0
            du_dx = self.gradient(u)
            adv = ops.multiply(ops.cast(self.velocity, self.compute_dtype), du_dx)
            residual = ops.subtract(adv, f)
        elif self.equation == "custom":
            try:
                residual = self.pde_fn(u, f)
            except TypeError:
                residual = self.pde_fn(u)

        if self.add_loss_enabled:
            loss = ops.cast(self.weight, self.compute_dtype) * ops.mean(ops.square(residual))
            self.add_loss(loss)

        if self.return_residual:
            return u, residual
        return u

    def compute_output_shape(self, input_shape: Any) -> Any:
        if isinstance(input_shape, (tuple, list)) and len(input_shape) == 2 and isinstance(input_shape[0], (tuple, list)):
            u_shape = input_shape[0]
        else:
            u_shape = input_shape

        if self.return_residual:
            return u_shape, u_shape
        return u_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "equation": self.equation,
                "dx": self.dx,
                "diffusivity": self.diffusivity,
                "nu": self.nu,
                "velocity": self.velocity,
                "forcing": self.forcing,
                "boundary_condition": self.boundary_condition,
                "weight": self.weight,
                "add_loss": self.add_loss_enabled,
                "return_residual": self.return_residual,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Residual":
        return cls(**config)


@keras.saving.register_keras_serializable(package="keras_sciml")
class PoissonResidual(Residual):
    """Poisson equation residual ($-\\Delta u - f = 0$) convenience layer."""

    def __init__(self, dx: Union[float, Sequence[float]] = 1.0, forcing: Union[float, Any] = 0.0, **kwargs: Any):
        kwargs.pop("equation", None)
        super().__init__(equation="poisson", dx=dx, forcing=forcing, **kwargs)


@keras.saving.register_keras_serializable(package="keras_sciml")
class DiffusionResidual(Residual):
    """Diffusion equation residual ($-\\alpha \\Delta u - f = 0$) convenience layer."""

    def __init__(
        self,
        diffusivity: float = 1.0,
        dx: Union[float, Sequence[float]] = 1.0,
        forcing: Union[float, Any] = 0.0,
        **kwargs: Any,
    ):
        kwargs.pop("equation", None)
        super().__init__(equation="diffusion", diffusivity=diffusivity, dx=dx, forcing=forcing, **kwargs)


@keras.saving.register_keras_serializable(package="keras_sciml")
class BurgersResidual(Residual):
    """Burgers equation residual ($u \\frac{\\partial u}{\\partial x} - \\nu \\Delta u - f = 0$) convenience layer."""

    def __init__(
        self,
        nu: float = 0.01,
        dx: Union[float, Sequence[float]] = 1.0,
        forcing: Union[float, Any] = 0.0,
        **kwargs: Any,
    ):
        kwargs.pop("equation", None)
        super().__init__(equation="burgers", nu=nu, dx=dx, forcing=forcing, **kwargs)


@keras.saving.register_keras_serializable(package="keras_sciml")
class AdvectionResidual(Residual):
    """Advection equation residual ($c \\frac{\\partial u}{\\partial x} - f = 0$) convenience layer."""

    def __init__(
        self,
        velocity: float = 1.0,
        dx: Union[float, Sequence[float]] = 1.0,
        forcing: Union[float, Any] = 0.0,
        **kwargs: Any,
    ):
        kwargs.pop("equation", None)
        super().__init__(equation="advection", velocity=velocity, dx=dx, forcing=forcing, **kwargs)
