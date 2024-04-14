import sys, os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)

import pytest
import tensorflow as tf
import numpy as np
from scripts.model.lgm_naive  import LgmSingleStepNaive  
from scripts.utils.utils.utils import ZeroBond, IRS, Swaption, TestExamples, Utils

@pytest.fixture
def setup_model_zerobond():
    T = 4
    n_steps = 12
    batch_size = 16
    setup_struct = {
        "T": T,
        "n_steps": n_steps,
        "batch_size": batch_size
    }
    X = tf.random.normal([n_steps * batch_size * T, 2], dtype = tf.float64)
    delta_x = tf.random.normal([n_steps * batch_size * T, 1], dtype = tf.float64)
    model = LgmSingleStepNaive(
        n_steps=tf.cast(n_steps * T, tf.float64),
        T=T,
        future_T=10,
        dim=2,
        verbose=False,
        sigma=0.01,
        batch_size=batch_size,
        phi=ZeroBond.Z_strike_normalized,
        name="zerobond",
        report_to_wandb=False,
        normalize=True,
        data_sample=X
    )
    model_not_normalize = LgmSingleStepNaive(
        n_steps=tf.cast(n_steps * T, tf.float64),
        T=T,
        future_T=10,
        dim=2,
        verbose=False,
        sigma=0.01,
        batch_size=batch_size,
        phi=ZeroBond.Z_strike_normalized,
        name="zerobond",
        report_to_wandb=False,
        normalize=False,
        data_sample=None
    )
    model.define_compiler(optimizer="adam", learning_rate=3e-5)
    model_not_normalize.define_compiler(optimizer="adam", learning_rate=3e-5)
    return model, model_not_normalize, X, delta_x, setup_struct

def test_compare_predicts(setup_model_zerobond):
    model, _, X, delta_x, _ = setup_model_zerobond
    results_tf = model.custom_train_step_tf(X, delta_x, False)
    results = model.custom_train_step(
        x = X,
        delta_x = delta_x,
        batch = 1,
        epoch = 1,
        apply_gradients = False
    )
    for i, value in enumerate(results_tf):
        assert np.allclose(value, results[i], atol=1e-6), f"{i} are not close enough"
        
def test_compare_predicts_not_normalize(setup_model_zerobond):
    _, model, X, delta_x, _ = setup_model_zerobond
    results_tf = model.custom_train_step_tf(X, delta_x, False)
    results = model.custom_train_step(
        x = X,
        delta_x = delta_x,
        batch = 1,
        epoch = 1,
        apply_gradients = False
    )
    for i, value in enumerate(results_tf):
        assert np.allclose(value, results[i], atol=1e-6), f"{i} are not close enough"