import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent  
sys.path.append(str(root_path))
# Imports
import tensorflow as tf
import numpy as np
# From
from tensorflow.keras import layers
from tensorflow import keras
# Utils
from scripts.utils.utils.utils import TFUtils
# typing
from typing import Any

class Losses():
    @staticmethod
    def get_normalized_loss(
        t1: tf.Tensor, 
        t2: tf.Tensor,
        L1: Any,
        L2: Any
    ) -> tf.Tensor:
        t1 = TFUtils.custom_reshape(
            t1
        )
        t2 = TFUtils.custom_reshape(
            t2
        )
        diff = (t1 - t2)
        diff_normalized = TFUtils.safe_divide(
            diff,
            tf.math.abs(t1)
        )
        partial_loss = tf.where(
            diff_normalized >= 1,
            L2(
                t1 / tf.math.abs(t1),
                t2 / tf.math.abs(t1)
            ),
            L1(diff_normalized) 
        )
        return partial_loss
    
    @staticmethod
    def get_loss(
        t1: tf.Tensor, 
        t2: tf.Tensor,
        L1: Any,
        L2: Any
    ) -> tf.Tensor:
        t1 = TFUtils.custom_reshape(
            t1
        )
        t2 = TFUtils.custom_reshape(
            t2
        )
        diff = (t1 - t2)
        partial_loss = tf.where(
            diff > 1,
            L2(
                t1,
                t2
            ),
            L1(diff) 
        )
        return partial_loss
    
    @staticmethod
    def loss_lgm( 
        x: tf.Tensor, 
        v: tf.Tensor, 
        ct: tf.Tensor,
        derivatives: tf.Tensor, 
        predictions: tf.Tensor, 
        N_steps: np.int64, 
        T: int = None,
        TM: int = None,
        phi: Any = None,
    ):
        # Loss functions
        L1 = tf.math.abs
        L2 = tf.math.squared_difference
        betas = [1.0, 1.0, 1.0]
        samples, _ = x.shape
        batch_size = int(np.floor(samples / N_steps))
        # For f and f'
        x_reformat = tf.reshape(x[:, 0], (batch_size, N_steps))
        # Loss given the strike function
        T = np.float64(T)
        TM = np.float64(TM)
        real_values = phi(
            xn = x_reformat[:, -1], 
            T = T,
            Tm = TM,
            ct = ct
        )
        # Strike loss
        strike_loss = Losses.get_loss(
            t1 = real_values,
            t2 = predictions[: , -1],
            L1 = L1,
            L2 = L2
        )
        strike_loss = tf.reduce_sum(strike_loss) / batch_size
        # Autodiff phi
        xn = tf.Variable(x_reformat[:, -1], name = 'xn', trainable = True)
        T = tf.Variable(np.float64(T), name = 'tn', trainable=False)
        TM = tf.Variable(np.float64(TM), name = 'ct', trainable=False)
        ct = tf.Variable(np.float64(ct), name = 'ct', trainable=False)
        # T = tf.Variable(np.float64(T), name = 'T', trainable=False)
        with tf.GradientTape() as tape:
            y = phi(
                xn = xn, 
                T = T, 
                Tm = TM, 
                ct = ct
            )
        grad_df = tape.gradient(
            y, 
            {
                'xn': xn   
            }
        )
        df_dxn = grad_df['xn'] if grad_df['xn'] is not None else 0. * xn
        # Derivative loss
        derivative_loss = Losses.get_loss(
            t1 = derivatives,
            t2 = df_dxn,
            L1 = L1,
            L2 = L2
        )
        derivative_loss = tf.reduce_sum(derivative_loss) / batch_size
        # Epoch error per step
        og_shape = v.shape
        v = tf.reshape(v, [-1])
        predictions = tf.reshape(predictions, [-1])
        step_loss = Losses.get_loss(
            t1 = v,
            t2 = predictions,
            L1 = L1,
            L2 = L2
        )
        step_loss = tf.reshape(step_loss, og_shape)
        error_per_step = tf.cumsum(
            step_loss[:, 1:],
            axis = 1
        )
        # Reduce sum the cumulative error
        error_per_step = tf.reduce_sum(
            error_per_step[:, -2] / N_steps / batch_size
        )
        # Weigth the errors
        strike_loss *= betas[0]
        derivative_loss *= betas[1]
        error_per_step *= betas[2]
        
        # Record internal losses
        losses_trackers = {
            't1': strike_loss,
            't2': derivative_loss,
            't3': error_per_step
        }
        
        loss_per_sample = tf.math.add(
            error_per_step, 
            tf.math.add(
                strike_loss, 
                derivative_loss
            )
        )
        return loss_per_sample, losses_trackers, df_dxn
