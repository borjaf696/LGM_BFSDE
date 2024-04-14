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
    n_steps = 38
    batch_size = 16
    setup_struct = {
        "T": T,
        "n_steps": n_steps,
        "batch_size": batch_size
    }
    model = LgmSingleStepNaive(
        n_steps=n_steps * T,
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
    X = tf.random.normal([n_steps * batch_size * T, 2], dtype = tf.float64)
    delta_x = tf.random.normal([n_steps * batch_size * T, 1], dtype = tf.float64)
    return model, X, delta_x, setup_struct

def test_predict_tf(setup_model_zerobond):
    model, X, delta_x, setup_struct = setup_model_zerobond
    v, predictions, grads_reshaped = model.predict_tf(X, delta_x, build_masks=False)
    
    T = setup_struct["T"]
    n_steps = setup_struct["n_steps"]
    batch_size = setup_struct["batch_size"]
    sample_expected = T * n_steps
    
    assert v.shape == (batch_size * sample_expected, 1)
    assert predictions.shape == (batch_size * sample_expected, 1)
    assert grads_reshaped.shape == (batch_size, sample_expected)

def test_predict(setup_model_zerobond):
    model, X, delta_x, setup_struct = setup_model_zerobond
    v, predictions, grads_reshaped = model.predict(X, delta_x, build_masks=False)
    
    T = setup_struct["T"]
    n_steps = setup_struct["n_steps"]
    batch_size = setup_struct["batch_size"]
    sample_expected = T * n_steps
    
    assert v.shape == (batch_size * sample_expected, 1)
    assert predictions.shape == (batch_size * sample_expected, 1)
    assert grads_reshaped.shape == (batch_size, sample_expected)
    
def test_compare_predicts(setup_model_zerobond):
    model, X, delta_x, _ = setup_model_zerobond
    v, predictions, grads_reshaped = model.predict(X, delta_x, build_masks=False)
    v_tf, predictions_tf, grads_reshaped_tf = model.predict_tf(X, delta_x, build_masks=False)
    
    assert np.allclose(v.numpy(), v_tf.numpy(), atol=1e-6), "V and V_tf are not close enough"
    assert np.allclose(predictions.numpy(), predictions_tf.numpy(), atol=1e-6), "Predictions and Predictions_tf are not close enough"
    assert np.allclose(grads_reshaped.numpy(), grads_reshaped_tf.numpy(), atol=1e-6), "Grads_reshaped and Grads_reshaped_tf are not close enough"