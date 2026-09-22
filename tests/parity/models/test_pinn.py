import numpy as np
import pytest
from keras import ops

from keras_sciml.models import PINN


def test_pinn_backbone_parity():
    # Verify PINN layer-by-layer forward pass matches direct composition
    model = PINN(output_dim=2, hidden_layers=3, hidden_units=16, activation="tanh")
    x = ops.ones((4, 2))

    out_model = ops.convert_to_numpy(model(x))

    # Sequential manual evaluation
    h = x
    for layer in model.dense_layers:
        h = layer(h)
    out_manual = ops.convert_to_numpy(model.out_layer(h))

    np.testing.assert_allclose(out_model, out_manual, rtol=1e-5, atol=1e-6)

