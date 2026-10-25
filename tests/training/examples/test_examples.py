import json
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"


def run_notebook(rel_path, **overrides):
    """Execute an example notebook's code cells in-process and return its namespace.

    The install/backend cell (any cell with a `!` shell line) is skipped so the test
    backend from `KERAS_BACKEND` is used. `overrides` replace the values set in the
    cell tagged `parameters`, so examples can be run small and fast.
    """
    cells = json.loads((EXAMPLES_DIR / rel_path).read_text())["cells"]
    namespace = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(line.startswith("!") for line in src.splitlines()):
            continue
        exec(compile(src, rel_path, "exec"), namespace)
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            unknown = set(overrides) - set(namespace)
            assert not unknown, f"Unknown parameters for {rel_path}: {unknown}"
            namespace.update(overrides)
    return namespace


def assert_trained(namespace, epochs):
    loss = namespace["history"].history["loss"]
    assert len(loss) == epochs
    assert loss[-1] < loss[0]


def test_example_fno_darcy():
    ns = run_notebook(
        "neural_operators/fno_darcy_flow.ipynb",
        N_SAMPLES=20,
        RESOLUTION=12,
        MODES=4,
        WIDTH=16,
        DEPTH=2,
        EPOCHS=10,
        BATCH_SIZE=10,
    )
    assert_trained(ns, 10)


def test_example_fno_burgers():
    ns = run_notebook(
        "neural_operators/fno_burgers.ipynb",
        N_SAMPLES=20,
        NX=20,
        MODES=4,
        WIDTH=16,
        DEPTH=2,
        EPOCHS=10,
        BATCH_SIZE=10,
    )
    assert_trained(ns, 10)


def test_example_deeponet():
    ns = run_notebook(
        "neural_operators/deeponet_antiderivative.ipynb",
        N_SAMPLES=20,
        M_SENSORS=15,
        N_POINTS=20,
        LATENT_DIM=16,
        HIDDEN_UNITS=32,
        EPOCHS=10,
        BATCH_SIZE=10,
    )
    assert_trained(ns, 10)


def test_example_pinn_poisson():
    ns = run_notebook(
        "physics_informed/pinn_poisson.ipynb",
        N_COLLOCATION=32,
        HIDDEN_LAYERS=2,
        HIDDEN_UNITS=32,
        EPOCHS=10,
    )
    assert_trained(ns, 10)


def test_example_pino_burgers():
    ns = run_notebook(
        "physics_informed/pino_burgers.ipynb",
        N_SAMPLES=16,
        NX=20,
        MODES=4,
        WIDTH=16,
        DEPTH=2,
        EPOCHS=10,
        BATCH_SIZE=8,
    )
    assert_trained(ns, 10)


def test_example_fno_sea_surface_temperature():
    ns = run_notebook(
        "climate/fno_sea_surface_temperature.ipynb",
        N_SAMPLES=16,
        RESOLUTION=16,
        TIME_HISTORY=1,
        TIME_FUTURE=1,
        MODES=4,
        WIDTH=16,
        DEPTH=2,
        EPOCHS=8,
        BATCH_SIZE=8,
        USE_SYNTHETIC=True,
    )
    assert_trained(ns, 8)
