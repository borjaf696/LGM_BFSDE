import pytest
from scripts.trainer.trainer import simulate
import numpy as np


@pytest.fixture
def simple_params():
    return {"T": 8, "N_steps": 100, "dim": 1, "sigma": 0.01, "nsims": 10000}


def test_simulation_simple(simple_params):
    df_x, _ = simulate(
        T=simple_params["T"],
        N_steps=simple_params["N_steps"],
        dim=simple_params["dim"],
        sigma=simple_params["sigma"],
        nsims=simple_params["nsims"],
    )

    means = (
        df_x.groupby(["dt"])
        .agg(mean_x=("X_0", "mean"), std_x=("X_0", "std"), num_cases=("X_0", "count"))
        .reset_index()
    )
    dts = means["dt"].values
    assert True
    avg_dt = 0
    n = means.shape[0] - 1
    for i in range(1, len(dts)):
        avg_dt += (dts[i] - dts[i - 1]) / n

    means["dummy_column"] = 1
    means["i"] = means["dummy_column"].cumsum() - 1
    means["expected_std"] = np.sqrt(avg_dt * means.i * simple_params["sigma"] ** 2)

    assert avg_dt - (simple_params["T"] / (simple_params["N_steps"] - 1)) < 1e-4
    assert (means.std_x - means.expected_std).mean() < 1e-3
    assert (means.mean_x - 0).mean() < 1e-3
