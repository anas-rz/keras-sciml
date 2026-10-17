import numpy as np
import pytest
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
torch = pytest.importorskip("torch")
import torch.nn as nn
import keras
from keras import ops

from keras_sciml.layers.operators import (
    SpectralConv1D,
    SpectralConv2D,
)


class TorchSpectralConv1d(nn.Module):
    """Reference SpectralConv1d from zongyi-li/fourier_neural_operator."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class TorchSpectralConv2d(nn.Module):
    """Reference SpectralConv2d from zongyi-li/fourier_neural_operator."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            torch.empty(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


def test_spectral_conv1d_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size = 2
    length = 32
    in_channels = 3
    out_channels = 4
    modes = 6

    # Generate test inputs and weights
    x_np = np.random.randn(batch_size, length, in_channels).astype("float32")
    w_r_np = np.random.randn(in_channels, out_channels, modes).astype("float32")
    w_i_np = np.random.randn(in_channels, out_channels, modes).astype("float32")

    # PyTorch reference (channels_first: B, in_c, length)
    x_torch = torch.from_numpy(np.transpose(x_np, (0, 2, 1)))
    torch_layer = TorchSpectralConv1d(in_channels, out_channels, modes)
    with torch.no_grad():
        torch_layer.weights1.copy_(
            torch.complex(torch.from_numpy(w_r_np), torch.from_numpy(w_i_np))
        )
    torch_out = torch_layer(x_torch).detach().numpy()
    torch_out_channels_last = np.transpose(torch_out, (0, 2, 1))

    # Keras implementation (channels_last: B, length, in_c)
    keras_layer = SpectralConv1D(filters=out_channels, modes=modes, use_bias=False)
    _ = keras_layer(ops.convert_to_tensor(x_np))
    keras_layer.set_weights([w_r_np, w_i_np])
    keras_out = ops.convert_to_numpy(keras_layer(ops.convert_to_tensor(x_np)))

    np.testing.assert_allclose(keras_out, torch_out_channels_last, atol=1e-5, rtol=1e-5)


def test_spectral_conv2d_parity_with_torch():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size = 2
    height, width = 16, 20
    in_channels = 3
    out_channels = 4
    modes1, modes2 = 4, 5

    x_np = np.random.randn(batch_size, height, width, in_channels).astype("float32")
    w1_r_np = np.random.randn(in_channels, out_channels, modes1, modes2).astype("float32")
    w1_i_np = np.random.randn(in_channels, out_channels, modes1, modes2).astype("float32")
    w2_r_np = np.random.randn(in_channels, out_channels, modes1, modes2).astype("float32")
    w2_i_np = np.random.randn(in_channels, out_channels, modes1, modes2).astype("float32")

    # PyTorch reference (channels_first: B, in_c, H, W)
    x_torch = torch.from_numpy(np.transpose(x_np, (0, 3, 1, 2)))
    torch_layer = TorchSpectralConv2d(in_channels, out_channels, modes1, modes2)
    with torch.no_grad():
        torch_layer.weights1.copy_(
            torch.complex(torch.from_numpy(w1_r_np), torch.from_numpy(w1_i_np))
        )
        torch_layer.weights2.copy_(
            torch.complex(torch.from_numpy(w2_r_np), torch.from_numpy(w2_i_np))
        )
    torch_out = torch_layer(x_torch).detach().numpy()
    torch_out_channels_last = np.transpose(torch_out, (0, 2, 3, 1))

    # Keras implementation (channels_last: B, H, W, in_c)
    keras_layer = SpectralConv2D(
        filters=out_channels, modes=(modes1, modes2), use_bias=False
    )
    _ = keras_layer(ops.convert_to_tensor(x_np))
    keras_layer.set_weights([w1_r_np, w1_i_np, w2_r_np, w2_i_np])
    keras_out = ops.convert_to_numpy(keras_layer(ops.convert_to_tensor(x_np)))

    np.testing.assert_allclose(keras_out, torch_out_channels_last, atol=1e-5, rtol=1e-5)

