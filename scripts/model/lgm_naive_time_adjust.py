import tensorflow as tf
import numpy as np

from scripts.model.model_lgm_single_step import LgmSingleStep


class LgmSingleStepNaiveTimeAdjust(LgmSingleStep):

    def predict(
        self,
        X: tf.Tensor,
        delta_x: tf.Tensor,
        build_masks: bool = False,
        debug: bool = False,
    ):
        """_summary_

        Args:
            X (tf.Tensor): _description_

        Returns:
            _type_: _description_
        """
        predictions = self._custom_model(X)
        if debug:
            print(f"Predictions shape: {predictions.shape}")
            print(f"Predictions: {predictions}")
        predictions = tf.cast(predictions, dtype=tf.float64)
        predictions_rolled = tf.roll(predictions, shift=1, axis=0)
        # Get the gradients X
        _, grads_x, _, grads_t = self._get_dv_dx(X)
        grads_x_rolled = tf.roll(grads_x, shift=1, axis=0)
        grads_t_rolled = tf.roll(grads_t, shift=1, axis=0)
        # Reshapes
        grads_x_rolled = tf.reshape(grads_x_rolled, (grads_x_rolled.shape[0], 1))
        grads_t_rolled = tf.reshape(grads_t_rolled, (grads_t_rolled.shape[0], 1))
        delta_x = tf.reshape(delta_x, (delta_x.shape[0], 1))
        # Calculate V
        v = tf.math.add(
            tf.math.add(predictions_rolled, tf.math.multiply(grads_x_rolled, delta_x)),
            grads_t_rolled * self._deltaT,
        )

        if not build_masks:
            mask_v = self._mask_v
            mask_preds = self._mask_preds
        else:
            # Create masks
            idx_preds = np.array(range(0, X.shape[0], self.N))
            np_mask_v = np.ones((X.shape[0], 1))
            np_mask_v[idx_preds] = 0
            mask_v = tf.convert_to_tensor(np_mask_v, dtype=tf.float64)
            mask_preds = tf.abs(tf.convert_to_tensor(np_mask_v, dtype=tf.float64) - 1)
        v = tf.math.add(
            tf.math.multiply(v, mask_v), tf.math.multiply(predictions, mask_preds)
        )
        # print(f'Error: {error}')
        return v, predictions
