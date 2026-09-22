from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops
from keras.losses import Loss

from keras_sciml.layers.differential.gradient import Gradient


@keras.saving.register_keras_serializable(package="keras_sciml")
class RelativeL2Loss(Loss):
    """Relative L2 error loss for operator learning and scientific machine learning.

    Computes the normalized relative error between predictions and ground truth:
    $$\\mathcal{L}(y, \\hat{y}) = \\left(\\frac{\\| y - \\hat{y} \\|_p}{\\| y \\|_p + \\epsilon}\\right)^q$$
    where $p=2$ by default and $q=2$ if `squared=True`, else $q=1$.

    This is the canonical evaluation and training loss used in Neural Operators (FNO, DeepONet, PINO).

    Args:
        p: float, norm degree (e.g. 2 for L2, 1 for L1). Defaults to 2.0.
        squared: bool, whether to square the relative error. Defaults to False.
        epsilon: float, numerical stability epsilon to prevent division by zero. Defaults to 1e-7.
        axes: int or tuple of ints, axes over which to calculate the norm for each sample.
            If `None`, automatically calculates norm over all dimensions except the batch dimension (dim 0).
        name: str, name of the loss. Defaults to `"relative_l2_loss"`.
        reduction: str, Keras loss reduction mode. Defaults to `"sum_over_batch_size"`.
    """

    def __init__(
        self,
        p: float = 2.0,
        squared: bool = False,
        epsilon: float = 1e-7,
        axes: Optional[Union[int, Sequence[int]]] = None,
        name: str = "relative_l2_loss",
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__(name=name, reduction=reduction)
        self.p = float(p)
        self.squared = bool(squared)
        self.epsilon = float(epsilon)
        if axes is not None:
            self.axes = (axes,) if isinstance(axes, int) else tuple(axes)
        else:
            self.axes = None

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_true = ops.cast(y_true, dtype=y_pred.dtype)
        diff = y_pred - y_true

        ndim = ops.ndim(y_true)
        if self.axes is not None:
            reduce_axes = self.axes
        elif ndim <= 1:
            reduce_axes = None
        else:
            reduce_axes = tuple(range(1, ndim))

        if reduce_axes is not None:
            if self.p == 2.0:
                num_sq = ops.sum(ops.square(diff), axis=reduce_axes)
                den_sq = ops.sum(ops.square(y_true), axis=reduce_axes)
                if self.squared:
                    return num_sq / (den_sq + self.epsilon)
                else:
                    return ops.sqrt(num_sq + self.epsilon) / (ops.sqrt(den_sq + self.epsilon) + self.epsilon)
            elif self.p == 1.0:
                num = ops.sum(ops.abs(diff), axis=reduce_axes)
                den = ops.sum(ops.abs(y_true), axis=reduce_axes) + self.epsilon
                res = num / den
                return ops.square(res) if self.squared else res
            else:
                num = ops.power(ops.sum(ops.power(ops.abs(diff), self.p), axis=reduce_axes) + self.epsilon, 1.0 / self.p)
                den = ops.power(ops.sum(ops.power(ops.abs(y_true), self.p), axis=reduce_axes) + self.epsilon, 1.0 / self.p)
                res = num / (den + self.epsilon)
                return ops.square(res) if self.squared else res
        else:
            if self.p == 2.0:
                if self.squared:
                    return ops.square(diff) / (ops.square(y_true) + self.epsilon)
                else:
                    return ops.abs(diff) / (ops.abs(y_true) + self.epsilon)
            else:
                res = ops.abs(diff) / (ops.abs(y_true) + self.epsilon)
                return ops.square(res) if self.squared else res

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "p": self.p,
            "squared": self.squared,
            "epsilon": self.epsilon,
            "axes": self.axes,
        }


@keras.saving.register_keras_serializable(package="keras_sciml")
class H1Loss(Loss):
    """Sobolev H1 / W^{1,2} loss penalizing function values and spatial derivatives.

    For fields on spatial grids, the H1 loss enforces both value agreement and derivative consistency:
    $$\\mathcal{L}_{H^1}(y, \\hat{y}) = \\| y - \\hat{y} \\|_2^2 + \\alpha \\| \\nabla(y - \\hat{y}) \\|_2^2$$
    or if `relative=True`:
    $$\\mathcal{L}_{H^1, \\text{rel}}(y, \\hat{y}) = \\frac{\\| y - \\hat{y} \\|_{H^1}}{\\| y \\|_{H^1} + \\epsilon}$$

    Args:
        alpha: float, weighting factor for the derivative gradient penalty. Defaults to 0.1.
        dx: float or sequence of floats, spatial grid step size. Defaults to 1.0.
        axes: int or sequence of ints, spatial axes along which to compute the gradient.
            If `None`, automatically infers spatial dimensions (e.g. `range(1, rank - 1)` for channel-last tensors).
        boundary_condition: str, finite difference boundary scheme (`"central"`, `"forward"`, `"periodic"`).
            Defaults to `"central"`.
        relative: bool, whether to compute relative H1 norm. Defaults to False.
        epsilon: float, numerical stability constant. Defaults to 1e-7.
        name: str, loss name. Defaults to `"h1_loss"`.
        reduction: str, Keras reduction mode. Defaults to `"sum_over_batch_size"`.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        dx: Union[float, Sequence[float]] = 1.0,
        axes: Optional[Union[int, Sequence[int]]] = None,
        boundary_condition: str = "central",
        relative: bool = False,
        epsilon: float = 1e-7,
        name: str = "h1_loss",
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__(name=name, reduction=reduction)
        self.alpha = float(alpha)
        self.dx = dx
        if axes is not None:
            self.axes = (axes,) if isinstance(axes, int) else tuple(axes)
        else:
            self.axes = None
        self.boundary_condition = boundary_condition
        self.relative = bool(relative)
        self.epsilon = float(epsilon)

        self._gradient_layer = Gradient(
            axes=self.axes,
            dx=self.dx,
            boundary_condition=self.boundary_condition,
        )

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_true = ops.cast(y_true, dtype=y_pred.dtype)
        diff = y_pred - y_true
        ndim = ops.ndim(y_true)

        # Compute gradient of the difference
        grad_diff = self._gradient_layer(diff)

        if not self.relative:
            # Standard Sobolev penalty: MSE(diff) + alpha * MSE(grad_diff)
            # Reduce across all spatial & channel dimensions per batch sample
            if ndim > 1:
                reduce_axes = tuple(range(1, ndim))
                l2_term = ops.mean(ops.square(diff), axis=reduce_axes)
                grad_reduce_axes = tuple(range(1, ops.ndim(grad_diff)))
                grad_term = ops.mean(ops.square(grad_diff), axis=grad_reduce_axes)
            else:
                l2_term = ops.square(diff)
                grad_term = ops.square(grad_diff)
            return l2_term + self.alpha * grad_term
        else:
            # Relative H1 norm
            if ndim > 1:
                reduce_axes = tuple(range(1, ndim))
                grad_reduce_axes = tuple(range(1, ops.ndim(grad_diff)))
                num_sq = ops.sum(ops.square(diff), axis=reduce_axes) + self.alpha * ops.sum(
                    ops.square(grad_diff), axis=grad_reduce_axes
                )

                grad_true = self._gradient_layer(y_true)
                den_sq = ops.sum(ops.square(y_true), axis=reduce_axes) + self.alpha * ops.sum(
                    ops.square(grad_true), axis=grad_reduce_axes
                )
                return ops.sqrt(num_sq + self.epsilon) / (ops.sqrt(den_sq + self.epsilon) + self.epsilon)
            else:
                num_sq = ops.square(diff) + self.alpha * ops.square(grad_diff)
                grad_true = self._gradient_layer(y_true)
                den_sq = ops.square(y_true) + self.alpha * ops.square(grad_true)
                return ops.sqrt(num_sq + self.epsilon) / (ops.sqrt(den_sq + self.epsilon) + self.epsilon)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "alpha": self.alpha,
            "dx": self.dx,
            "axes": self.axes,
            "boundary_condition": self.boundary_condition,
            "relative": self.relative,
            "epsilon": self.epsilon,
        }


@keras.saving.register_keras_serializable(package="keras_sciml")
class PhysicsLoss(Loss):
    """Composite physics loss combining data loss and optional PDE residual loss.

    $$\\mathcal{L} = w_{\\text{data}} \\mathcal{L}_{\\text{data}}(y, \\hat{y}) + w_{\\text{pde}} \\mathcal{L}_{\\text{pde}}(\\hat{y})$$

    Supported `loss_type`:
    - `"relative_l2"`: Relative L2 norm error.
    - `"h1"`: Sobolev H1 gradient-aware loss.
    - `"mse"`: Standard mean squared error.
    - `"mae"`: Mean absolute error.

    Args:
        loss_type: str, type of data loss (`"relative_l2"`, `"h1"`, `"mse"`, `"mae"`). Defaults to `"relative_l2"`.
        data_weight: float, weight for the data loss term. Defaults to 1.0.
        pde_weight: float, weight for the PDE residual term. Defaults to 0.0.
        residual_fn: Optional callable computing PDE residual $R(y_pred)$. Defaults to None.
        dx: float or sequence of floats, grid spacing (used for `"h1"`). Defaults to 1.0.
        alpha: float, Sobolev H1 derivative weight (used for `"h1"`). Defaults to 0.1.
        epsilon: float, numerical stability epsilon. Defaults to 1e-7.
        name: str, loss name. Defaults to `"physics_loss"`.
        reduction: str, Keras loss reduction mode. Defaults to `"sum_over_batch_size"`.
    """

    def __init__(
        self,
        loss_type: str = "relative_l2",
        data_weight: float = 1.0,
        pde_weight: float = 0.0,
        residual_fn: Optional[Callable[[Any], Any]] = None,
        dx: Union[float, Sequence[float]] = 1.0,
        alpha: float = 0.1,
        epsilon: float = 1e-7,
        name: str = "physics_loss",
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__(name=name, reduction=reduction)
        loss_type = loss_type.lower()
        if loss_type not in ("relative_l2", "h1", "mse", "mae"):
            raise ValueError(f"`loss_type` must be one of ('relative_l2', 'h1', 'mse', 'mae'), got: {loss_type}")

        self.loss_type = loss_type
        self.data_weight = float(data_weight)
        self.pde_weight = float(pde_weight)
        self.residual_fn = residual_fn
        self.dx = dx
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

        if self.loss_type == "relative_l2":
            self._base_loss: Optional[Loss] = RelativeL2Loss(epsilon=self.epsilon, reduction="none")
        elif self.loss_type == "h1":
            self._base_loss = H1Loss(alpha=self.alpha, dx=self.dx, epsilon=self.epsilon, reduction="none")
        else:
            self._base_loss = None

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_true = ops.cast(y_true, dtype=y_pred.dtype)
        if self._base_loss is not None:
            data_loss = self._base_loss(y_true, y_pred)
        else:
            diff = y_pred - y_true
            ndim = ops.ndim(diff)
            reduce_axes = tuple(range(1, ndim)) if ndim > 1 else None
            if self.loss_type == "mse":
                data_loss = ops.mean(ops.square(diff), axis=reduce_axes) if reduce_axes is not None else ops.square(diff)
            else:  # mae
                data_loss = ops.mean(ops.abs(diff), axis=reduce_axes) if reduce_axes is not None else ops.abs(diff)

        loss = self.data_weight * data_loss

        if self.pde_weight > 0.0 and self.residual_fn is not None:
            pde_res = self.residual_fn(y_pred)
            ndim = ops.ndim(pde_res)
            reduce_axes = tuple(range(1, ndim)) if ndim > 1 else None
            pde_loss = ops.mean(ops.square(pde_res), axis=reduce_axes) if reduce_axes is not None else ops.square(pde_res)
            loss = loss + self.pde_weight * pde_loss
        return loss

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "loss_type": self.loss_type,
            "data_weight": self.data_weight,
            "pde_weight": self.pde_weight,
            "dx": self.dx,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
        }


def relative_l2_error(y_true: Any, y_pred: Any, epsilon: float = 1e-7) -> Any:
    """Convenience function computing per-sample relative L2 error."""
    loss_fn = RelativeL2Loss(epsilon=epsilon, reduction="none")
    return loss_fn(y_true, y_pred)

