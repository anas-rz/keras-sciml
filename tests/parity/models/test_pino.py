import numpy as np
import pytest
from keras import ops

from keras_sciml.models import PINO, FNO1D


def test_pino_fno_equivalence_parity():
    # Verify PINO prediction identically matches wrapped FNO backbone
    backbone = FNO1D(modes=4, width=8, depth=2, out_channels=1)
    pino = PINO(operator_model=backbone, pde_weight=0.1)

    x = ops.ones((2, 16, 1))

    # In eval/inference mode (training=False), predictions must be identical
    out_backbone = ops.convert_to_numpy(backbone(x))
    out_pino = ops.convert_to_numpy(pino(x, training=False))

    np.testing.assert_allclose(out_pino, out_backbone, rtol=1e-5, atol=1e-6)

