import sys, os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)

import pytest
import numpy as np
import tensorflow as tf
from scripts.utils.utils.utils import ZeroBond, IRS, Swaption, TestExamples, Utils
from scripts.losses.losses import Losses

@pytest.fixture
def loss_lgm_tf_fixture():
    batch_size = tf.constant(16, dtype = tf.int32)
    N_steps = tf.constant(48, dtype = tf.float64)
    T_val = tf.constant(4, dtype = tf.float64)
    TM_val = tf.constant(8, dtype = tf.float64)
    ct = tf.constant(0.9, dtype = tf.float64)
    
    sample_size = batch_size * tf.cast(N_steps * T_val, dtype = tf.int32)
    tensor_shape = (sample_size, 1)
    tensor_predictions = (batch_size, tf.cast(N_steps * T_val, dtype = tf.int32))
    tensor_shape_derivatives = (batch_size, 1)

    x = tf.random.normal(tensor_shape, dtype = tf.float64)
    derivatives = tf.random.normal(tensor_shape_derivatives, dtype = tf.float64)
    v = tf.random.normal(tensor_predictions, dtype = tf.float64)
    predictions = tf.random.normal(tensor_predictions, dtype = tf.float64)

    T = T_val
    TM = TM_val
    phi=ZeroBond.Z_strike_normalized

    args = {
        'x': x,
        'v': v,
        'ct': ct,
        'derivatives': derivatives,
        'predictions': predictions,
        'N_steps': N_steps * T,
        'T': T,
        'TM': TM,
        'batch_size': batch_size,
        'phi': phi,
    }

    return args

def test_loss_lgm_tf(loss_lgm_tf_fixture):
    result = Losses.loss_lgm_tf(**loss_lgm_tf_fixture)

    assert result is not None
    
def test_loss_lgm(loss_lgm_tf_fixture):
    result = Losses.loss_lgm(**loss_lgm_tf_fixture)

    assert result is not None
    
def test_loss_compare(loss_lgm_tf_fixture):
    result_tf, _, _ = Losses.loss_lgm_tf(**loss_lgm_tf_fixture)
    result, _, _ = Losses.loss_lgm(**loss_lgm_tf_fixture)

    assert np.allclose(result.numpy(), result_tf.numpy(), atol=1e-6), "V and V_tf are not close enough"