# Imports
import tensorflow as tf
import numpy as np
# From
from tensorflow.keras import layers
from tensorflow import keras
# Utils
from utils.utils.utils import TFUtils
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
        max_per_row = tf.reduce_max(
            tf.concat(
                [
                    tf.math.abs(t1),
                    tf.math.abs(t2)
                ],
                axis = 1
            ), 
            axis = 1
        )
        max_per_row = TFUtils.custom_reshape(
            max_per_row
        )
        diff_normalized = TFUtils.safe_divide(
            diff,
            max_per_row
        )
        partial_loss = tf.where(
            diff_normalized > 1,
            L2(
                t1 / max_per_row,
                t2 / max_per_row
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
        # Strike loss
        strike_loss = Losses.get_normalized_loss(
            t1 = real_values,
            t2 = predictions[: , -1],
            L1 = L1,
            L2 = L2
        )
        strike_loss = tf.reshape(
            strike_loss,
            (batch_size, 1)
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
        # Derivative loss
        derivative_loss = Losses.get_normalized_loss(
            t1 = derivatives,
            t2 = df_dxn,
            L1 = L1,
            L2 = L2
        )
        derivative_loss = tf.reshape(
            derivative_loss,
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
        # Epoch error per step
        og_shape = v.shape
        v = tf.reshape(v, [-1])
        predictions = tf.reshape(predictions, [-1])
        step_loss = Losses.get_normalized_loss(
            t1 = v,
            t2 = predictions,
            L1 = L1,
            L2 = L2
        )
        step_loss = tf.reshape(step_loss, og_shape)
        error_per_step = tf.cumsum(
            step_loss,
            axis = 1
        ) / (N_steps)
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
        # Apply mask to only change given the last step
        if mask_loss is not None:
            loss_per_sample = tf.math.multiply(loss_per_sample, mask_loss)
        '''import sys
        idx_preds = np.array(range(N_steps - 1, strike_loss_reshaped.shape[0], N_steps))
        print(f'Loss per sample: {loss_per_sample.numpy()[idx_preds]}')
        print(f'Idx preds: {idx_preds}')
        print(f'Mean loss: {loss_per_sample.numpy()[idx_preds].mean()}')
        sys.exit()'''
        return loss_per_sample, losses_trackers, df_dxn

    @staticmethod
    def loss_lgm_agg( 
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
        # Strike loss
        strike_loss = Losses.get_normalized_loss(
            t1 = real_values,
            t2 = predictions[: , -1],
            L1 = L1,
            L2 = L2
        )
        strike_loss = tf.reshape(
            strike_loss,
            (batch_size, 1)
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
        # Derivative loss
        derivative_loss = Losses.get_normalized_loss(
            t1 = derivatives,
            t2 = df_dxn,
            L1 = L1,
            L2 = L2
        )
        derivative_loss = tf.reshape(
            derivative_loss,
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
        # Epoch error per step
        og_shape = v.shape
        v = tf.reshape(v, [-1])
        predictions = tf.reshape(predictions, [-1])
        step_loss = Losses.get_normalized_loss(
            t1 = v,
            t2 = predictions,
            L1 = L1,
            L2 = L2
        )
        step_loss = tf.reshape(step_loss, og_shape)
        error_per_step = tf.cumsum(
            step_loss,
            axis = 1
        ) / (N_steps)
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
        # Apply mask to only change given the last step
        if mask_loss is not None:
            loss_per_sample = tf.math.multiply(loss_per_sample, mask_loss)
        '''import sys
        idx_preds = np.array(range(N_steps - 1, strike_loss_reshaped.shape[0], N_steps))
        print(f'Loss per sample: {loss_per_sample.numpy()[idx_preds]}')
        print(f'Idx preds: {idx_preds}')
        print(f'Mean loss: {loss_per_sample.numpy()[idx_preds].mean()}')
        sys.exit()'''
        return loss_per_sample, losses_trackers, df_dxn
