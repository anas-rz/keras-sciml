import numpy as np
import pytest
from keras import ops

from keras_sciml.models import FNO1D


def test_fno1d_manual_decomposition_parity():
    # Verify FNOBlock1D computes exactly: act(SpectralConv1D(x) + Dense(x))
    model = FNO1D(modes=4, width=8, depth=1, out_channels=1, projection_dim=16, activation="gelu")
    x = ops.ones((2, 16, 1))

    # Run forward pass to build weights
    out_model = ops.convert_to_numpy(model(x))

    # Manually compute layer by layer
    v0 = model.lifting(x)
    block = model.fno_blocks[0]
    spec_out = block.spectral_conv(v0)
    bypass_out = block.bypass(v0)
    v1 = block.act(spec_out + bypass_out)
    proj1 = model.proj1(v1)
    out_manual = ops.convert_to_numpy(model.proj2(proj1))

    np.testing.assert_allclose(out_model, out_manual, rtol=1e-5, atol=1e-6)

