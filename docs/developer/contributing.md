# Contributing to keras-sciml

Contributions to `keras-sciml` are welcome! Whether you are adding a new neural operator architecture, differential layer, or benchmark dataset, please follow these guidelines.

---

## 1. Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/anas-raza/keras-sciml.git
   cd keras-sciml
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate
   ```

3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[test,docs]"
   ```

4. Install your target backend (e.g. `torch`, `tensorflow`, or `jax[cpu]`).

---

## 2. Multi-Backend Guidelines

`keras-sciml` maintains strict backend agnosticism. When writing layers or models:

* **Use `keras.ops` exclusively**: Never import backend-specific functions directly (e.g. avoid `torch.*`, `tf.*`, or `jax.numpy.*` in core package code).
* **Do not assume tensor storage layout**: Always support standard Keras channel ordering (`channels_last`).
* **Handle serialization**: Implement `get_config()` and `classmethod from_config(config)` for all custom layers and models.
* **Test on all 3 backends**: Ensure all new tests pass under `KERAS_BACKEND=torch`, `KERAS_BACKEND=tensorflow`, and `KERAS_BACKEND=jax`.

---

## 3. Submitting Pull Requests

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/my-new-operator
   ```

2. Add corresponding **smoke**, **parity**, and **training** tests in `tests/`.

3. Ensure documentation builds without errors:
   ```bash
   mkdocs build --strict
   ```

4. Push your branch and open a Pull Request. CI will automatically run tests across all backends if Python files are modified.

