# Imports
import tensorflow as tf
import numpy as np
# From
from tensorflow.keras import layers
from tensorflow import keras
# typing
from typing import Any

class Losses():
    @staticmethod
    def loss_lgm( 
                 x: tf.Tensor, 
                 v: tf.Tensor, 
                 ct: tf.Tensor,
                 derivatives: tf.Tensor, 
                 predictions: tf.Tensor, 
                 N_steps: np.int64, 
                 verbose: bool = False,
                 T: int = None,
                 TM: int = None,
                 phi: Any = None,
                 mask_loss: tf.Tensor = None):
        
        # print(f'Predictions:{predictions[0,:]}, V: {v[0, :]}')
        betas = [0.02, 0.02, 1]
        # Tiles
        tile_multiples = tf.constant([1, N_steps], tf.int64)
        samples, _ = x.shape
        batch_size = int(np.floor(samples / N_steps))
        # For f and f'
        x_reformat = tf.reshape(x[:, 0], (batch_size, N_steps))
        xn_tensor = x_reformat[:, -1]
        # Loss given the strike function
        tn = np.float64(T)
        TM = np.float64(TM)
        strike_loss = tf.reshape(
            tf.math.squared_difference(
                predictions[:, -1], 
                phi(
                    xn_tensor, 
                    tn,
                    TM,
                    ct)
                ), 
            (batch_size,1)
        )
        
        # Repeat the tensor to adapt dimensions
        strike_loss = tf.tile(
            strike_loss,
            tile_multiples
        )
        strike_loss_reshaped = tf.reshape(strike_loss, [-1])
        # Autodiff f
        xn = tf.Variable(xn_tensor, name = 'xn', trainable = True)
        tn = tf.Variable(np.float64(T), name = 'tn', trainable=False)
        ct = tf.Variable(np.float64(ct), name = 'ct', trainable=False)
        with tf.GradientTape() as tape:
            y = phi(
                xn, 
                tn, 
                T, 
                ct
            )
        grad_df = tape.gradient(y, {
            'xn':xn   
        })
        df_dxn = grad_df['xn'] if grad_df['xn'] is not None else 0. * xn
        # Verbose to output
        if verbose:
            log_file = 'logs/20230217/grads_fx.log'
            with open(log_file, 'a+') as f:
                vector = x[:, -1]
                f.write(f'X:\n')
                for x_i in vector:
                    f.write(f'{x_i},')
                f.write(f'\n')
                f.write(f'Grads: \n')
                for grad_i in vector:
                    f.write(f'{grad_i},')
                f.write(f'\n')
        # Careful: global variable
        derivative_loss = tf.reshape(
            tf.math.squared_difference(
                derivatives,
                df_dxn
            ), 
            (batch_size, 1))
        # Repeat the tensor to adapt dimensions
        derivative_loss = tf.tile(
            derivative_loss,
            [1, N_steps]
        )
        derivative_loss_reshaped = tf.reshape(derivative_loss, [-1])
        # TODO: Check if this is correct
        # Epoch error per step
        error_per_step = tf.cumsum(
            tf.math.squared_difference(
                v, 
                predictions
            ), axis = 1) / (N_steps)
        # Flatten the cumsum
        error_per_step = tf.reshape(error_per_step, [-1])
        # Record internal losses
        losses_trackers = {
            't1': strike_loss,
            't2': derivative_loss,
            't3': error_per_step
        }
        # Weigth the errors
        strike_loss *= betas[0]
        derivative_loss *= betas[1]
        error_per_step *= betas[2]
        
        loss_per_sample = tf.math.add(
            error_per_step, 
            tf.math.add(
                strike_loss_reshaped, 
                derivative_loss_reshaped
            )
        )
        '''loss_per_sample = tf.math.add(
            error_per_step, 
            strike_loss_reshaped
        )'''
        # Sanity purposes
        difference_strike = tf.math.reduce_mean(
            tf.math.abs(
                v[:, -1] - phi(xn_tensor,tn,TM,ct)
            )
        )
        # Apply mask to only change given the last step
        if mask_loss is not None:
            loss_per_sample = tf.math.multiply(loss_per_sample, mask_loss)
        return loss_per_sample, losses_trackers, df_dxn, difference_strike