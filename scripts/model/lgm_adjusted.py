import tensorflow as tf
import numpy as np
import wandb

from scripts.losses.losses import Losses
from scripts.model.model_lgm_single_step import LgmSingleStep


class LgmSingleStepModelAdjusted(LgmSingleStep):
    """The adjusted model only uses Y_0 to calculate the rest of the values.

    Args:
        LgmSingleStep (_type_): _description_
    """

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
        # Get the gradients
        _, grads_x, _, _ = self._get_dv_dx(X)
        # Reshapes
        grads_x_reshaped = tf.reshape(grads_x, (self._batch_size, self.N))
        predictions_reshaped = tf.reshape(predictions, (self._batch_size, self.N))
        delta_x_reshaped = tf.reshape(delta_x, (self._batch_size, self.N))
        v = np.zeros((self._batch_size, self.N))
        v[:, 0] = predictions_reshaped[:, 0]
        for i in range(1, self.N):
            v[:, i] = v[:, i - 1] + grads_x_reshaped[:, i - 1] * delta_x_reshaped[:, i]
        v = tf.convert_to_tensor(np.reshape(v, (self._batch_size * self.N, 1)))
        return v, predictions
