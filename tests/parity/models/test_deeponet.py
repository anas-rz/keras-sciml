import numpy as np
import pytest
from keras import ops

from keras_sciml.models import DeepONet


def test_deeponet_dot_product_parity():
    # Verify DeepONet computes exact mathematical formula: sum_k b_k(u) * t_k(y) + b0
    model = DeepONet(latent_dim=8, hidden_layers=2, hidden_units=16, mode="cartesian", use_bias=True)
    u = ops.ones((3, 10))
    y = ops.ones((20, 2))

    out_model = ops.convert_to_numpy(model([u, y]))

    # Manually compute branch and trunk
    b_feat = ops.convert_to_numpy(model.branch_net(u))  # (3, 8)
    t_feat = ops.convert_to_numpy(model.trunk_net(y))   # (20, 8)
    bias = ops.convert_to_numpy(model.merger.bias)      # (1,)

    out_manual = np.matmul(b_feat, t_feat.T) + bias     # (3, 20)

    np.testing.assert_allclose(out_model, out_manual, rtol=1e-5, atol=1e-6)

