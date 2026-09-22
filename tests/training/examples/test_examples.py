import pytest

from examples.deeponet_antiderivative import run_experiment as run_deeponet_example
from examples.fno_burgers import run_experiment as run_fno_burgers_example
from examples.fno_darcy_flow import run_experiment as run_fno_darcy_example
from examples.pinn_poisson import run_experiment as run_pinn_example
from examples.pino_burgers import run_experiment as run_pino_example
from examples.fno_sea_surface_temperature import run_experiment as run_fno_sst_example


def test_example_fno_darcy():
    model, history, test_loss = run_fno_darcy_example(
        n_samples=20,
        resolution=12,
        modes=4,
        width=16,
        depth=2,
        epochs=10,
        batch_size=10,
        verbose=0,
    )
    assert len(history.history["loss"]) == 10
    assert history.history["loss"][-1] < history.history["loss"][0]


def test_example_fno_burgers():
    model, history, test_loss = run_fno_burgers_example(
        n_samples=20,
        nx=20,
        modes=4,
        width=16,
        depth=2,
        epochs=10,
        batch_size=10,
        verbose=0,
    )
    assert len(history.history["loss"]) == 10
    assert history.history["loss"][-1] < history.history["loss"][0]


def test_example_deeponet():
    model, history, test_loss = run_deeponet_example(
        n_samples=20,
        m_sensors=15,
        n_points=20,
        latent_dim=16,
        hidden_units=32,
        epochs=10,
        batch_size=10,
        verbose=0,
    )
    assert len(history.history["loss"]) == 10
    assert history.history["loss"][-1] < history.history["loss"][0]


def test_example_pinn_poisson():
    model, history, l2_error = run_pinn_example(
        n_collocation=32,
        hidden_layers=2,
        hidden_units=32,
        epochs=10,
        verbose=0,
    )
    assert len(history.history["loss"]) == 10
    assert history.history["loss"][-1] < history.history["loss"][0]


def test_example_pino_burgers():
    model, history, final_loss = run_pino_example(
        n_samples=16,
        nx=20,
        modes=4,
        width=16,
        depth=2,
        epochs=10,
        batch_size=8,
        verbose=0,
    )
    assert len(history.history["loss"]) == 10
    assert history.history["loss"][-1] < history.history["loss"][0]


def test_example_fno_sea_surface_temperature():
    model, history, test_rel_l2 = run_fno_sst_example(
        n_samples=16,
        resolution=16,
        time_history=1,
        time_future=1,
        modes=4,
        width=16,
        depth=2,
        epochs=8,
        batch_size=8,
        use_synthetic=True,
        verbose=0,
    )
    assert len(history.history["loss"]) == 8
    assert history.history["loss"][-1] < history.history["loss"][0]


