import numpy as np
import pytest

torch = pytest.importorskip("torch")
import keras
from keras import ops

from keras_sciml.layers.operators import (
    DeepONetCartesianProduct,
    DeepONetPointwise,
)


def test_cartesian_parity_against_numpy_and_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size = 8
    num_points = 15
    latent_p = 16

    b_np = np.random.randn(batch_size, latent_p).astype("float32")
    t_np = np.random.randn(num_points, latent_p).astype("float32")
    bias_np = np.array([0.42], dtype="float32")

    # Numpy reference
    ref_np = np.matmul(b_np, t_np.T) + bias_np[0]

    # PyTorch reference
    b_torch = torch.from_numpy(b_np)
    t_torch = torch.from_numpy(t_np)
    bias_torch = torch.from_numpy(bias_np)
    ref_torch = (torch.matmul(b_torch, t_torch.T) + bias_torch).numpy()

    # Keras implementation
    layer = DeepONetCartesianProduct(use_bias=True)
    _ = layer([ops.convert_to_tensor(b_np), ops.convert_to_tensor(t_np)])
    layer.set_weights([bias_np])
    keras_out = ops.convert_to_numpy(
        layer([ops.convert_to_tensor(b_np), ops.convert_to_tensor(t_np)])
    )

    np.testing.assert_allclose(keras_out, ref_np, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(keras_out, ref_torch, atol=1e-6, rtol=1e-5)


def test_batched_cartesian_parity_against_numpy_and_torch():
    np.random.seed(123)
    torch.manual_seed(123)

    batch_size = 4
    num_points = 10
    latent_p = 8

    b_np = np.random.randn(batch_size, latent_p).astype("float32")
    t_np = np.random.randn(batch_size, num_points, latent_p).astype("float32")
    bias_np = np.array([-0.15], dtype="float32")

    # Numpy reference
    ref_np = np.einsum("bp,bnp->bn", b_np, t_np) + bias_np[0]

    # PyTorch reference
    b_torch = torch.from_numpy(b_np)
    t_torch = torch.from_numpy(t_np)
    ref_torch = (torch.einsum("bp,bnp->bn", b_torch, t_torch) + bias_np[0]).numpy()

    # Keras implementation
    layer = DeepONetCartesianProduct(use_bias=True)
    _ = layer([ops.convert_to_tensor(b_np), ops.convert_to_tensor(t_np)])
    layer.set_weights([bias_np])
    keras_out = ops.convert_to_numpy(
        layer([ops.convert_to_tensor(b_np), ops.convert_to_tensor(t_np)])
    )

    np.testing.assert_allclose(keras_out, ref_np, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(keras_out, ref_torch, atol=1e-6, rtol=1e-5)


def test_pointwise_parity_against_numpy_and_torch():
    np.random.seed(99)
    torch.manual_seed(99)

    batch_size = 12
    latent_p = 10

    b_np = np.random.randn(batch_size, latent_p).astype("float32")
    t_np = np.random.randn(batch_size, latent_p).astype("float32")
    bias_np = np.array([0.75], dtype="float32")

    ref_np = np.sum(b_np * t_np, axis=-1, keepdims=True) + bias_np[0]

    b_torch = torch.from_numpy(b_np)
    t_torch = torch.from_numpy(t_np)
    ref_torch = (torch.sum(b_torch * t_torch, dim=-1, keepdim=True) + bias_np[0]).numpy()

    layer = DeepONetPointwise(use_bias=True)
    _ = layer([ops.convert_to_tensor(b_np), ops.convert_to_tensor(t_np)])
    layer.set_weights([bias_np])
    keras_out = ops.convert_to_numpy(
        layer([ops.convert_to_tensor(b_np), ops.convert_to_tensor(t_np)])
    )

    np.testing.assert_allclose(keras_out, ref_np, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(keras_out, ref_torch, atol=1e-6, rtol=1e-5)

