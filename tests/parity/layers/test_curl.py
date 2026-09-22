import numpy as np
import pytest
import keras
from keras import ops

from keras_sciml.layers.differential import Curl, Divergence, Gradient


def test_parity_against_numpy_gradient_2d():
    np.random.seed(42)
    batch_size, h, w = 2, 14, 16
    dx = 0.1
    dy = 0.08
    v_np = np.random.randn(batch_size, h, w, 2).astype("float32")

    ref_dvy_dx = np.gradient(v_np[:, :, :, 1], dx, axis=1)
    ref_dvx_dy = np.gradient(v_np[:, :, :, 0], dy, axis=2)
    ref_curl = (ref_dvy_dx - ref_dvx_dy)[..., None]

    layer = Curl(dx=[dx, dy], boundary_condition="central", keepdims=True)
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    np.testing.assert_allclose(keras_out, ref_curl, atol=1e-5, rtol=1e-5)


def test_parity_against_numpy_gradient_3d():
    np.random.seed(999)
    batch_size, d, h, w = 1, 8, 10, 12
    spacings = [0.2, 0.1, 0.05]
    v_np = np.random.randn(batch_size, d, h, w, 3).astype("float32")

    vx = v_np[..., 0]
    vy = v_np[..., 1]
    vz = v_np[..., 2]

    ref_curl_x = np.gradient(vz, spacings[1], axis=2) - np.gradient(vy, spacings[2], axis=3)
    ref_curl_y = np.gradient(vx, spacings[2], axis=3) - np.gradient(vz, spacings[0], axis=1)
    ref_curl_z = np.gradient(vy, spacings[0], axis=1) - np.gradient(vx, spacings[1], axis=2)
    ref_curl = np.stack([ref_curl_x, ref_curl_y, ref_curl_z], axis=-1)

    layer = Curl(dx=spacings, boundary_condition="central")
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    np.testing.assert_allclose(keras_out, ref_curl, atol=1e-5, rtol=1e-5)


def test_parity_2d_rigid_body_rotation():
    # v = (-omega * y, omega * x) -> curl = 2 * omega everywhere
    omega = 2.5
    h = 0.05
    n = 20
    coords = np.linspace(-1.0, 1.0, n, dtype="float32")
    xx, yy = np.meshgrid(coords, coords, indexing="ij")

    vx = -omega * yy
    vy = omega * xx
    v_np = np.stack([vx, vy], axis=-1)[None, ...]

    dx = float(coords[1] - coords[0])
    layer = Curl(dx=dx, boundary_condition="central")
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    # Central difference is exact for linear functions
    np.testing.assert_allclose(keras_out, 2.0 * omega, atol=1e-5)


def test_parity_3d_solid_body_rotation():
    # v = omega x r -> curl = 2 * omega everywhere
    omega = np.array([1.5, -2.0, 3.0], dtype="float32")
    n = 12
    coords = np.linspace(-1.0, 1.0, n, dtype="float32")
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")

    # v = (omega_y * z - omega_z * y, omega_z * x - omega_x * z, omega_x * y - omega_y * x)
    vx = omega[1] * zz - omega[2] * yy
    vy = omega[2] * xx - omega[0] * zz
    vz = omega[0] * yy - omega[1] * xx
    v_np = np.stack([vx, vy, vz], axis=-1)[None, ...]

    h = float(coords[1] - coords[0])
    layer = Curl(dx=h, boundary_condition="central")
    keras_out = ops.convert_to_numpy(layer(ops.convert_to_tensor(v_np)))

    expected_curl = np.broadcast_to((2.0 * omega).reshape(1, 1, 1, 1, 3), keras_out.shape)
    np.testing.assert_allclose(keras_out, expected_curl, atol=1e-5)


def test_parity_irrotational_curl_of_gradient_is_zero():
    # Identity: curl(grad(phi)) = 0 for any smooth scalar field
    n = 24
    h = 1.0 / n
    coords = np.linspace(0.0, 1.0, n, endpoint=False, dtype="float32")
    xx, yy = np.meshgrid(coords, coords, indexing="ij")

    phi = np.sin(2.0 * np.pi * xx) * np.cos(4.0 * np.pi * yy)
    phi_tensor = ops.convert_to_tensor(phi[None, ..., None])

    grad_layer = Gradient(dx=h, boundary_condition="periodic", concat_components=True)
    grad_phi = grad_layer(phi_tensor)

    curl_layer = Curl(dx=h, boundary_condition="periodic")
    curl_grad = ops.convert_to_numpy(curl_layer(grad_phi))

    np.testing.assert_allclose(curl_grad, 0.0, atol=5e-5)



def test_parity_solenoidal_divergence_of_curl_is_zero_3d():
    # Identity: div(curl(v)) = 0 for any smooth 3D vector field
    n = 16
    h = 1.0 / n
    coords = np.linspace(0.0, 1.0, n, endpoint=False, dtype="float32")
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")

    vx = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    vy = np.cos(2.0 * np.pi * yy) * np.sin(2.0 * np.pi * zz)
    vz = np.sin(2.0 * np.pi * zz) * np.cos(2.0 * np.pi * xx)
    v_np = np.stack([vx, vy, vz], axis=-1)[None, ...]

    v_tensor = ops.convert_to_tensor(v_np)
    curl_layer = Curl(dx=h, boundary_condition="periodic")
    curl_v = curl_layer(v_tensor)

    div_layer = Divergence(dx=h, boundary_condition="periodic")
    div_curl = ops.convert_to_numpy(div_layer(curl_v))

    np.testing.assert_allclose(div_curl, 0.0, atol=1e-5)
