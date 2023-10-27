import pytest
import tensorflow as tf
from scripts.model.lgm_naive import LgmSingleStepNaive
from scripts.utils.utils.utils import MathUtils

@pytest.fixture
def params():
    return {
        "N_steps": 10,
        "T": 8,
        "future_T": None,
        "dim": 2,
        "sigma": 0.01,
        "batch_size": 64,
        "phi": None,
        "phi_str": "zerobond",
        "report_to_wandb": False,
        "normalize": False,
        "data_sample": None
    }
    
@pytest.fixture
def x():
    return tf.random.normal([1000*10, 2], mean=0, stddev=5)

def test_model_gradients_respect_x(params, x):
    n, _ = x.shape
    lgm = LgmSingleStepNaive(
            n_steps = params["N_steps"], 
            T = params["T"], 
            future_T = params["future_T"],
            dim = params["dim"],
            verbose = False,
            sigma = params["sigma"],
            batch_size = params["batch_size"],
            phi = params["phi"],
            name = params["phi_str"],
            report_to_wandb = params["report_to_wandb"],
            normalize = params["normalize"],
            data_sample= params["data_sample"]
        )
    _, grads_x, _, grads_t = lgm._get_dv_dx(
        x
    )
    custom_grads = MathUtils.central_difference_gradients(
        x,
        lgm._custom_model
    )
    custom_grads_x = custom_grads[:, 0]
    custom_grads_t = custom_grads[:, 1]
    
    assert (tf.reduce_sum(custom_grads_x - grads_x)/n < 1e-3) &\
        (tf.reduce_sum(custom_grads_t - grads_t)/n < 1e-3)