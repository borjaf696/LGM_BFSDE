import tensorflow as tf
import numpy as np

from scripts.model.model_lgm_single_step import LgmSingleStep

class LgmSingleStepNaive(LgmSingleStep):
    
    def predict(self, X:tf.Tensor, 
                delta_x:tf.Tensor,
                build_masks: bool = False,
                debug: bool = False):
        """_summary_

        Args:
            X (tf.Tensor): _description_

        Returns:
            _type_: _description_
        """
        predictions = self._custom_model(X)
        if debug:
            print(f'Predictions shape: {predictions.shape}')
            print(f'Predictions: {predictions}')
        predictions = tf.cast(
            predictions, 
            dtype=tf.float64
        )
        predictions_rolled = tf.roll(
            predictions,
            shift = 1,
            axis = 0
        )
        # Get the gradients
        grads = self._get_dv_dx(X)[1]
        grads_rolled = tf.roll(
            grads,
            shift = 1,
            axis = 0
        )
        # Reshapes
        grads_rolled = tf.reshape(grads, (grads.shape[0], 1))
        delta_x = tf.reshape(delta_x, (delta_x.shape[0], 1))
        # Calculate V
        v = tf.math.add(
            predictions_rolled,
            tf.math.multiply(
                grads_rolled,
                delta_x
            )
        )
        # Sanity
        '''grads_reshaped = tf.reshape(grads, (self._batch_size, self.N))
        predictions_reshaped = tf.reshape(predictions, (self._batch_size, self.N))
        delta_x_reshaped = tf.reshape(delta_x, (self._batch_size, self.N))
        v_sanity = np.zeros((self._batch_size, self.N))
        v_sanity[:, 0] = predictions_reshaped[:, 0]
        for i in range(1, self.N):
            v_sanity[:, i] = predictions_reshaped[:, i - 1] + grads_reshaped[:, i - 1] * delta_x_reshaped[:, i]
        v_sanity = tf.convert_to_tensor(
            np.reshape(
                v_sanity, 
                (self._batch_size * self.N, 1)
            )   
        )'''
        '''print(f'{v.shape}, {predictions_rolled.shape}, {grads_rolled.shape}')
        print(f'{self._mask_v.shape}, {self._mask_preds.shape}')'''
        if not build_masks:
            mask_v = self._mask_v
            mask_preds = self._mask_preds
        else:
            # Create masks
            idx_preds = np.array(range(0, X.shape[0], self.N))
            np_mask_v = np.ones((X.shape[0], 1))
            np_mask_v[idx_preds] = 0
            mask_v = tf.convert_to_tensor(
                np_mask_v, 
                dtype = tf.float64
            )
            mask_preds = tf.abs(
                tf.convert_to_tensor(
                    np_mask_v,
                    dtype = tf.float64
                ) - 1)
        v = tf.math.add(
            tf.math.multiply(
                v,
                mask_v
            ),
            tf.math.multiply(
                predictions,
                mask_preds
            )
        )
        '''error = tf.math.reduce_sum(
            tf.math.subtract(
                v_sanity,
                v
            )
        )'''
        # print(f'Error: {error}')
        return v, predictions
