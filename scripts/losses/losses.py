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
        mask_loss: tf.Tensor = None
    ):
        # Loss functions
        L1 = tf.math.abs
        L2 = tf.math.squared_difference
        # print(f'Predictions:{predictions[0,:]}, V: {v[0, :]}')
        betas = [1.0, 1.0, 1.0]
        # Tiles
        tile_multiples = tf.constant([1, N_steps], tf.int64)
        samples, _ = x.shape
        batch_size = int(np.floor(samples / N_steps))
        # Final shape
        final_shape = [
            N_steps * batch_size,
            1
        ]
        # For f and f'
        x_reformat = tf.reshape(x[:, 0], (batch_size, N_steps))
        xn_tensor = x_reformat[:, -1]
        # Loss given the strike function
        tn = np.float64(T)
        TM = np.float64(TM)
        real_values = phi(
            xn_tensor, 
            tn,
            TM,
            ct
        )
        diff = predictions[:, -1] - real_values
        strike_loss = tf.reshape(
            tf.where(
                diff > 1,
                L2(
                    predictions[:, -1],
                    real_values
                ),
                L1(diff) 
            ),
            (batch_size,1)
        )
        '''print(f'V: {v[:10, -1]}')
        print(f'Predictions: {predictions[:10, -1]}')
        print(f'Real values: {real_values[:10]}')
        print(f'Errors: {tf.math.squared_difference(predictions[:10, -1], real_values[:10])}')'''
        # Repeat the tensor to adapt dimensions
        strike_loss = tf.tile(
            strike_loss,
            tile_multiples
        )
        strike_loss_reshaped = tf.reshape(
            strike_loss, 
            final_shape
        )
        # Autodiff f
        xn = tf.Variable(xn_tensor, name = 'xn', trainable = True)
        tn = tf.Variable(np.float64(T), name = 'tn', trainable=False)
        ct = tf.Variable(np.float64(ct), name = 'ct', trainable=False)
        # T = tf.Variable(np.float64(T), name = 'T', trainable=False)
        with tf.GradientTape() as tape:
            y = phi(
                xn, 
                tn, 
                T, 
                ct
            )
        grad_df = tape.gradient(
            y, 
            {
                'xn':xn   
            }
        )
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
        diff = derivatives - df_dxn
        derivative_loss = tf.reshape(
            tf.where(
                diff > 1,
                L2(
                    derivatives,
                    df_dxn
                ),
                L1(diff) 
            ),
            (batch_size, 1)
        )
        # Repeat the tensor to adapt dimensions
        derivative_loss = tf.tile(
            derivative_loss,
            tile_multiples
        )
        derivative_loss_reshaped = tf.reshape(
            derivative_loss, 
            final_shape
        )
        # TODO: Check if this is correct
        # Epoch error per step
        diff = v - predictions
        error_per_step = tf.cumsum(
            tf.where(
                diff > 1,
                L2(
                    v,
                    predictions
                ),
                L1(diff) 
            ),  axis = 1) / (N_steps)
        # Flatten the cumsum
        error_per_step = tf.reshape(
            error_per_step, 
            final_shape
        )
        # Record internal losses
        losses_trackers = {
            't1': strike_loss_reshaped,
            't2': derivative_loss_reshaped,
            't3': error_per_step
        }
        # Weigth the errors
        strike_loss_reshaped *= betas[0]
        derivative_loss_reshaped *= betas[1]
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
        '''import sys
        idx_preds = np.array(range(N_steps - 1, strike_loss_reshaped.shape[0], N_steps))
        print(f'Loss per sample: {loss_per_sample.numpy()[idx_preds]}')
        print(f'Idx preds: {idx_preds}')
        print(f'Mean loss: {loss_per_sample.numpy()[idx_preds].mean()}')
        sys.exit()'''
        return loss_per_sample, losses_trackers, df_dxn, difference_strike
