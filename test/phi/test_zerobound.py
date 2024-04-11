import pytest
import tensorflow as tf
import pandas as pd
import numpy as np
from scripts.utils.utils.utils import ZeroBond, FinanceUtils
from scripts.trainer.trainer import simulate

columns = ["X_0", "delta_x_0", "dt", "sim"]
T = 8
nsteps = 1000


@pytest.fixture
def simple_params():
    return {"T": T, "N_steps": nsteps, "dim": 1, "sigma": 0.01, "nsims": 100}


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


def test_zerobound_one_year(data):
    T = int(data.dt.max())
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
    ct = tf.Variable(np.float64(ct), name="ct", trainable=False)
    T = tf.Variable(np.float64(T), name="T", trainable=False)
    # Fix the batch size
    batch_size = int(xt.shape[0] / nsteps)
    # Real values
    v_real = ZeroBond.Z_tensor(xt, dt, T, ct)
    v_real_reshaped = tf.reshape(v_real, (batch_size, nsteps))
    n_tensor = ZeroBond.N_tensor(dt, xt, ct)
    # Derivative:
    xt = tf.Variable(xt, name="xn", trainable=True)
    dt = tf.Variable(dt, name="tn", trainable=False)
    ct = tf.Variable(np.float64(ct), name="ct", trainable=False)
    with tf.GradientTape() as tape:
        y = ZeroBond.Z_normalized(xt, dt, T, ct)
    grad_df = tape.gradient(y, {"xn": xt})
    grads = grad_df["xn"]
    # Simulate - LGM step:
    grads_reshaped = tf.reshape(grads, (batch_size, nsteps))
    xt_reshaped = tf.reshape(xt, (batch_size, nsteps))
    delta_x_reshaped = tf.reshape(delta_x, (batch_size, nsteps))
    # Calculate the MVP
    v = np.ones((batch_size, nsteps)) * np.float64(v_real_reshaped[0, 0])
    for i in range(1, nsteps):
        v[:, i] = v[:, i - 1] + grads_reshaped[:, i - 1] * delta_x_reshaped[:, i]
    # Calculate errors absolute
    print(
        f"Xt: {xt.shape[0]} batch size: {batch_size} Reshaped: {tf.reshape(v, -1).shape[0]}"
    )
    v_real = np.array(tf.reshape(v_real_reshaped, -1))
    v_column = np.array(tf.reshape(v, -1)) * n_tensor.numpy()
    dt_list = np.array(dt)
    df_results = pd.DataFrame(
        zip(xt.numpy(), v_real, v_column, dt_list, n_tensor.numpy()),
        columns=["xt", "v_real", "v_est", "dt", "n"],
    )
    # Error
    df_results["absolute_error"] = (df_results.v_real - df_results.v_est).abs()

    assert df_results.absolute_error.mean() < 1e-2
