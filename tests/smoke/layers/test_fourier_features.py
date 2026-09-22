import tempfile
import pytest
import keras
from keras import ops
import numpy as np

from keras_sciml.layers.operators import FourierFeatures


def test_instantiation():
    layer = FourierFeatures(output_dim=64, scale=5.0)
    assert layer.output_dim == 64
    assert layer.scale == 5.0
    assert not layer.trainable_frequencies
    assert not layer.include_input

    with pytest.raises(ValueError, match="even positive integer"):
        FourierFeatures(output_dim=63)

    with pytest.raises(ValueError, match="even positive integer"):
        FourierFeatures(output_dim=0)


def test_shapes_2d_input():
    layer = FourierFeatures(output_dim=32, scale=1.0)
    x = ops.ones((4, 3))
    y = layer(x)
    assert y.shape == (4, 32)
    assert layer.compute_output_shape((4, 3)) == (4, 32)


def test_shapes_nd_input():
    layer = FourierFeatures(output_dim=16, scale=1.0)
    x = ops.ones((2, 8, 10, 4))
    y = layer(x)
    assert y.shape == (2, 8, 10, 16)
    assert layer.compute_output_shape((2, 8, 10, 4)) == (2, 8, 10, 16)


def test_include_input():
    layer = FourierFeatures(output_dim=16, include_input=True)
    x = ops.ones((5, 3))
    y = layer(x)
    assert y.shape == (5, 19)  # 16 + 3
    assert layer.compute_output_shape((5, 3)) == (5, 19)
    # Check that first 3 features are equal to inputs
    np.testing.assert_allclose(ops.convert_to_numpy(y[:, :3]), 1.0, atol=1e-6)


def test_trainable_vs_frozen_weights():
    frozen_layer = FourierFeatures(output_dim=16, trainable_frequencies=False)
    frozen_layer(ops.ones((2, 4)))
    assert len(frozen_layer.trainable_weights) == 0
    assert len(frozen_layer.non_trainable_weights) == 1

    trainable_layer = FourierFeatures(output_dim=16, trainable_frequencies=True)
    trainable_layer(ops.ones((2, 4)))
    assert len(trainable_layer.trainable_weights) == 1
    assert len(trainable_layer.non_trainable_weights) == 0


def test_serialization_config():
    layer = FourierFeatures(
        output_dim=32,
        scale=2.5,
        trainable_frequencies=True,
        include_input=True,
        two_pi=False,
    )
    config = layer.get_config()
    reconstructed = FourierFeatures.from_config(config)

    assert reconstructed.output_dim == 32
    assert reconstructed.scale == 2.5
    assert reconstructed.trainable_frequencies is True
    assert reconstructed.include_input is True
    assert reconstructed.two_pi is False


def test_model_save_and_load():
    inputs = keras.layers.Input(shape=(2,))
    x = FourierFeatures(output_dim=16, scale=1.0)(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    test_input = np.random.uniform(size=(3, 2)).astype("float32")
    original_pred = model(test_input)

    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        model.save(tmp.name)
        loaded_model = keras.models.load_model(tmp.name)
        loaded_pred = loaded_model(test_input)

    np.testing.assert_allclose(
        ops.convert_to_numpy(original_pred),
        ops.convert_to_numpy(loaded_pred),
        atol=1e-5,
    )

