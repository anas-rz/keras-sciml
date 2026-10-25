# Installation

`keras-sciml` is designed to work with Python 3.10+ and Keras 3.

---

## 1. Prerequisites & Backend Selection

Keras 3 supports three computational backends: **PyTorch**, **TensorFlow**, and **JAX**. Before installing `keras-sciml`, install your preferred backend:

=== "PyTorch"
    ```bash
    # For CPU
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # For CUDA GPU (e.g. CUDA 12.1)
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

=== "TensorFlow"
    ```bash
    # For CPU
    pip install tensorflow-cpu

    # For GPU
    pip install tensorflow
    ```

=== "JAX"
    ```bash
    # For CPU
    pip install "jax[cpu]"

    # For CUDA GPU
    pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

---

## 2. Installing keras-sciml

Install the latest release from PyPI:

```bash
pip install keras-sciml
```

Or install from source for development:

```bash
git clone https://github.com/anas-rz/keras-sciml.git
cd keras-sciml
pip install -e .
```

### Optional Dependencies

For running the test suite:
```bash
pip install -e ".[test]"
```

For building documentation locally:
```bash
pip install -e ".[docs]"
```

---

## 3. Selecting Your Active Backend

You can select the active backend by setting the `KERAS_BACKEND` environment variable before importing Keras:

=== "Bash / Linux / macOS"
    ```bash
    export KERAS_BACKEND="torch"        # Use PyTorch
    # or
    export KERAS_BACKEND="tensorflow"   # Use TensorFlow
    # or
    export KERAS_BACKEND="jax"          # Use JAX
    ```

=== "Python Inline"
    ```python
    import os
    os.environ["KERAS_BACKEND"] = "torch"  # Must be set before importing keras

    import keras
    print("Active backend:", keras.backend.backend())
    ```

=== "Configuration File"
    You can also edit `~/.keras/keras.json`:
    ```json
    {
        "floatx": "float32",
        "epsilon": 1e-07,
        "backend": "torch",
        "image_data_format": "channels_last"
    }
    ```

