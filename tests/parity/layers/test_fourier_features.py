import math
import numpy as np
import pytest
import torch
import keras
from keras import ops

from keras_sciml.layers.operators import FourierFeatures


class TorchReferenceFourierFeatures(torch.nn.Module):
    """Reference PyTorch implementation following Tancik et al. (2020)."""

    def __init__(self, weight: torch.Tensor, two_pi: bool = True, include_input: bool = False):
        super().__init__()
        self.weight = weight
        self.two_pi = two_pi
        self.include_input = include_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.two_pi:
            x_proj = 2.0 * math.pi * x
        else:
            x_proj = x
        proj = torch.matmul(x_proj, self.weight)
        features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        if self.include_input:
            return torch.cat([x, features], dim=-1)
        return features


def test_parity_against_reference_and_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    in_dim = 3
    num_frequencies = 8
    output_dim = 2 * num_frequencies
    batch_size = 16

    # Fixed weights
    weight_np = np.random.normal(0.0, 1.0, size=(in_dim, num_frequencies)).astype("float32")
    x_np = np.random.uniform(-1.0, 1.0, size=(batch_size, in_dim)).astype("float32")

    # Reference numpy output
    proj_np = 2.0 * np.pi * np.matmul(x_np, weight_np)
    ref_np_out = np.concatenate([np.cos(proj_np), np.sin(proj_np)], axis=-1)

    # Reference PyTorch output
    torch_layer = TorchReferenceFourierFeatures(
        weight=torch.from_numpy(weight_np),
        two_pi=True,
        include_input=False,
    )
    torch_out = torch_layer(torch.from_numpy(x_np)).detach().cpu().numpy()

    # Keras implementation
    keras_layer = FourierFeatures(
        output_dim=output_dim,
        two_pi=True,
        include_input=False,
    )
    # Build and set weight
    _ = keras_layer(ops.convert_to_tensor(x_np))
    keras_layer.set_weights([weight_np])
    keras_out = ops.convert_to_numpy(keras_layer(ops.convert_to_tensor(x_np)))

    # Verify parity against both numpy and torch reference
    np.testing.assert_allclose(keras_out, ref_np_out, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(keras_out, torch_out, rtol=1e-5, atol=1e-6)


def test_parity_with_include_input_and_no_two_pi():
    np.random.seed(123)

    in_dim = 2
    num_frequencies = 4
    output_dim = 2 * num_frequencies
    x_np = np.random.uniform(-2.0, 2.0, size=(10, 5, in_dim)).astype("float32")
    weight_np = np.random.normal(0.0, 2.0, size=(in_dim, num_frequencies)).astype("float32")

    torch_layer = TorchReferenceFourierFeatures(
        weight=torch.from_numpy(weight_np),
        two_pi=False,
        include_input=True,
    )
    torch_out = torch_layer(torch.from_numpy(x_np)).detach().cpu().numpy()

    keras_layer = FourierFeatures(
        output_dim=output_dim,
        two_pi=False,
        include_input=True,
    )
    _ = keras_layer(ops.convert_to_tensor(x_np))
    keras_layer.set_weights([weight_np])
    keras_out = ops.convert_to_numpy(keras_layer(ops.convert_to_tensor(x_np)))

    np.testing.assert_allclose(keras_out, torch_out, rtol=1e-5, atol=1e-6)

