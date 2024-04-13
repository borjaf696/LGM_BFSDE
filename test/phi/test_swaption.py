import sys, os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)

import pytest
import tensorflow as tf
import pandas as pd
import numpy as np
from scripts.utils.utils.utils import Swaption, ZeroBond, FinanceUtils
from scripts.trainer.trainer import simulate

T = 8
Tm = 10
nsteps_per_year = 48
nsteps = T * nsteps_per_year
columns = ["X_0", "delta_x_0", "dt", "sim"]


@pytest.fixture
def simple_params():
    return {"T": T, "Tm": Tm, "N_steps": nsteps, "dim": 1, "sigma": 0.01, "nsims": 10}


@pytest.fixture
def data(simple_params):
    data, _ = simulate(
        T=simple_params["T"],
        N_steps=simple_params["N_steps"],
        dim=simple_params["dim"],
        sigma=simple_params["sigma"],
        nsims=simple_params["nsims"],
    )
    return data[columns]


def test_irs(data, simple_params):
    T = simple_params["T"]
    Tm = simple_params["Tm"]
    # Get data to process the ZeroBound
    delta_x = data.delta_x_0.values
    xt = data.X_0.values
    dt = data.dt.values
    t_unique = data.dt.unique()
    dict_C = {dt: FinanceUtils.C(dt, sigma_value=0.01) for dt in t_unique}
    ct = data.apply(lambda x: dict_C[x["dt"]], axis=1)
    nt = ZeroBond.N_tensor(dt, xt, ct)
    # Convert to tensors
    xt = tf.convert_to_tensor(xt, dtype=tf.float64)
    delta_x = tf.convert_to_tensor(delta_x, dtype=tf.float64)
    dt = tf.convert_to_tensor(dt, dtype=tf.float64)
    ct = tf.convert_to_tensor(ct, dtype=tf.float64)
    T = tf.convert_to_tensor(T, dtype=tf.float64)
    Tm = tf.convert_to_tensor(Tm, dtype=tf.float64)
    # Fix the batch size
    batch_size = int(xt.shape[0] / nsteps)
    # Real values
    v_real = Swaption.Swaption_test(xn=xt, t=dt, Ti=T, Tm=Tm, ct=ct)
    v_real_reshaped = tf.reshape(v_real, (batch_size, nsteps))
    n_tensor = ZeroBond.N_tensor(dt, xt, ct)
    # Derivative:
    xt = tf.Variable(xt, name="xn", trainable=True)
    dt = tf.Variable(dt, name="tn", trainable=False)
    ct = tf.Variable(np.float64(ct), name="ct", trainable=False)
    T = tf.Variable(np.float64(T), name="T", trainable=False)
    Tm = tf.Variable(np.float64(Tm), name="Tm", trainable=False)
    with tf.GradientTape() as tape:
        y = Swaption.Swaption_test_normalized(xn=xt, t=dt, Ti=T, Tm=Tm, ct=ct)
    grad_df = tape.gradient(y, {"xn": xt})
    grads = grad_df["xn"]
    # Simulate - LGM step:
    grads_reshaped = tf.reshape(grads, (batch_size, nsteps))
    delta_x_reshaped = tf.reshape(delta_x, (batch_size, nsteps))
    # Calculate the MVP
    v = np.ones((batch_size, nsteps)) * np.float64(v_real_reshaped[0, 0])
    for i in range(1, nsteps):
        v[:, i] = v[:, i - 1] + grads_reshaped[:, i - 1] * delta_x_reshaped[:, i]
    # Calculate errors absolute
    v_real = np.array(tf.reshape(v_real_reshaped, -1))
    v_column = np.array(tf.reshape(v, -1)) * n_tensor.numpy()
    dt_list = np.array(dt)
    df_results = pd.DataFrame(
        zip(xt.numpy(), v_real, v_column, dt_list, n_tensor.numpy(), grads.numpy()),
        columns=["xt", "v_real", "v_est", "dt", "n", "grads"],
    )
    # Error
    df_results["absolute_error"] = (df_results.v_real - df_results.v_est).abs()

    assert df_results.absolute_error.mean() < 1e-2
