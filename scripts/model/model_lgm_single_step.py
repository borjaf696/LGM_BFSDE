import sys
from pathlib import Path
import os
import psutil
root_path = Path(__file__).resolve().parent.parent.parent  
sys.path.append(str(root_path))
# Sys
import sys
import os
import gc
# Load configuration
import json
# Imports
import tensorflow as tf
import numpy as np
import time
# From
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
# Utils
from scripts.utils.utils.utils import (
    FinanceUtils
) 
# Loss functions
from scripts.losses.losses import Losses
# Wandb
import wandb

# TODO: Store all the metadata
class LgmSingleStep(tf.keras.Model):

    def __init__(
        self,
        n_steps,
        T = 0,
        name="zerobond",
        *,
        dim = 2,
        verbose = False,
        sigma = None,
        batch_size = None,
        phi = None,
        future_T = None,
        report_to_wandb = False,
        normalize = False,
        # First simulation
        data_sample = None,
        **kwargs
    ):
        """_summary_

        Args:
            n_steps (_type_): _description_
            T (int, optional): _description_. Defaults to 0.
            name (str, optional): _description_. Defaults to "LGM_NN_model_single_step".
            verbose (bool, optional): _description_. Defaults to False.
            sigma (_type_, optional): _description_. Defaults to None.
            batch_size (_type_, optional): _description_. Defaults to None.
            phi (_type_, optional): _description_. Defaults to None.
        """
        super(LgmSingleStep, self).__init__(name=name, **kwargs)
        # Training relevant attributes
        self.N = n_steps
        self.T = np.float64(T)
        # Same for actives with out future
        self.future_T = future_T
        self._batch_size = batch_size
        self._expected_sample_size = self.N * self._batch_size
        # Phi function
        self.__phi = phi
        # Type of active
        self.__name = name
        # Constantes:
        print(f"{'#'*100}")
        print(f'Strike time (T): {T}')
        print(f'Second strike time (TM): {self.future_T}')
        print(f'Number of steps per path: {n_steps}')
        print(f"Model name: {self.__name}")
        print(f"Batch size: {self._batch_size}")
        print(f"Expected sample size: {self._expected_sample_size}")
        print(f"{'#'*100}")
        # Model with time and value
        input_tmp = keras.Input(
            shape = (dim, ), 
            name = self.__name
        )
        if normalize:
            # Normalizer
            start_normalization_time = time.time()
            normalizer = preprocessing.Normalization()
            normalizer.adapt(
                data_sample
            )
            x = normalizer(
                input_tmp
            )
            end_normalization_time = time.time()
            print(f"Normalization time: {end_normalization_time - start_normalization_time}s")
        else:
            x = input_tmp
        # Configuration read from:
        # --- name
        # --- T, strike time
        configuration = None
        with open('scripts/configs/ff_config.json', 'r+') as f:
            configuration = json.load(f)[name][str(T)]
        self.__num_layers = configuration['layers']
        print(f'Number of layers: {self.__num_layers}')
        print(f'Number of hidden units: {configuration["hidden_units"]}')
        # Build dense layers
        self.__dense_layers = []
        self.__batch_norm_layers = []

        for i in range(self.__num_layers):
            self.__dense_layers.append(
                tf.keras.layers.Dense(
                    units = configuration['hidden_units'],
                    kernel_initializer = tf.keras.initializers.HeUniform(),
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    name = 'internal_dense_'+str(i)
                )
            )
            self.__batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        for i in range(0, self.__num_layers):
            x = self.__dense_layers[i](x)
            x = self.__batch_norm_layers[i](x)
            x = tf.keras.layers.ReLU()(x)
        output_tmp = layers.Dense(
            units = 1, 
            kernel_initializer = tf.keras.initializers.GlorotUniform(),
            name = 'output_dense')(x)
        self._custom_model = keras.Model(
            inputs = input_tmp,
            outputs = output_tmp,
            name = name)
        # Set the loss function
        self.loss_lgm = Losses.loss_lgm
        # Metrics tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # Internal management
        self._loss_tracker_t1 = tf.keras.metrics.Mean(name="strike_loss")
        self._loss_tracker_t2 = tf.keras.metrics.Mean(name="derivative_loss")
        self._loss_tracker_t3 = tf.keras.metrics.Mean(name="step_loss")
        # Duration each step
        self._deltaT = T / self.N
        # CT
        self._ct = FinanceUtils.C(T, sigma_value = sigma)
        # Status variables
        self._grads, self._predictions = None, None
        # Create masks
        self.__create_masks()
        # Verbose
        self._verbose = verbose
        # Track with wandb
        self.__wandb = report_to_wandb
        
    def __create_masks(self):
        # Create masks
        idx_preds = np.array(range(0, self._expected_sample_size, self.N))
        mask_v = np.ones((self._expected_sample_size, 1))
        mask_v[idx_preds] = 0
        self._mask_v = tf.convert_to_tensor(
            (mask_v == 1), 
            dtype = tf.float64
        )
        self._mask_preds = tf.convert_to_tensor(
            (mask_v == 0),
            dtype = tf.float64
        )
        print(f'Positions to avoid from V {self._expected_sample_size - np.sum(self._mask_v)}')
        print(f'Positions to complete from V {np.sum(self._mask_preds)}')
        
        # Loss mask
        print(f'Loss mask: {self._expected_sample_size / self.N}')
        idx_preds = np.array(range(self.N - 1, self._expected_sample_size + 1, self.N))
        mask_loss = np.zeros((self._expected_sample_size, 1)) 
        mask_loss[idx_preds] = 1.0
        self._mask_loss = tf.reshape(
            tf.convert_to_tensor(
                (mask_loss == 1),
                dtype = tf.float64
            ),
            (self._expected_sample_size, 1)
        )
        print(f'Positions to avoid from loss {self._expected_sample_size - np.sum(self._mask_loss)}')
        print(f'Positions to complete from loss {np.sum(self._mask_loss)}')
        # Size of the masks
        number_of_elements = tf.size(self._mask_loss).numpy() + tf.size(self._mask_preds).numpy()
        element_size = self._mask_loss.dtype.size
        print(f'[MEMORY] Total memory consumed by masks: {number_of_elements * element_size / 2**30} Gb')
        
        # Remove extra data
        del mask_loss, idx_preds, mask_v
    @property
    def model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._custom_model
    
    @property
    def metrics(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return [self.loss_tracker, self.mae_metric]
    
    def summary(self):
        """_summary_
        """
        self._custom_model.summary()

    def call(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # TODO: Complete the delta_x
        return self.predict(inputs)
    
    @tf.function
    def train_step(self, data):
        pass
    
    def define_compiler(self, optimizer = 'adam', learning_rate = 1e-3):
        """_summary_

        Args:
            optimizer (str, optional): _description_. Defaults to 'adam'.
            learning_rate (_type_, optional): _description_. Defaults to 1e-3.
        """
        if optimizer == 'adam':
            print(f'Optimizer set to {optimizer}')
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
    def learning_rate(self, learning_rate):
        self._optimizer.learning_rate.assign(learning_rate)
    
    def custom_train_step(
        self, 
        x: tf.Tensor,
        batch: int, 
        epoch: int, 
        delta_x: tf.Tensor
    ):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            epoch (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        import os, psutil
        process = psutil.Process(os.getpid())
        memory_use = process.memory_info().rss / (1024 * 1024)
        print(f"\tMemory usage_1(grad): {memory_use}")
        batch_size = tf.shape(x)[0] // self.N
        with tf.GradientTape(persistent = False) as tape:
            memory_use = process.memory_info().rss / (1024 * 1024)
            print(f"\tMemory usage_1(pred): {memory_use}")
            v, predictions = self.predict(x, delta_x=delta_x)
            memory_use = process.memory_info().rss / (1024 * 1024)
            print(f"\tMemory usage_2(pred): {memory_use}")
            v = tf.reshape(v, (batch_size, self.N))
            predictions = tf.reshape(predictions, (batch_size, self.N))
            memory_use = process.memory_info().rss / (1024 * 1024)
            print(f"\tMemory usage_1(loss): {memory_use}")
            loss_values, losses_tracker, analytical_grads = self.loss_lgm(
                x=x,
                v=v,
                ct=self._ct,
                derivatives=self._get_dv_dxi(self.N - 1),
                predictions=predictions,
                N_steps=self.N,
                T=self.T,
                TM=self.future_T,
                phi=self.__phi
            )
            memory_use = process.memory_info().rss / (1024 * 1024)
            print(f"\tMemory usage_2(loss): {memory_use}")
        grads = tape.gradient(loss_values, self.model.trainable_weights)
        process = psutil.Process(os.getpid())
        memory_use = process.memory_info().rss / (1024 * 1024)
        print(f"\tMemory usage_2 (grad): {memory_use}")
        gc.collect()
        memory_use = process.memory_info().rss / (1024 * 1024)
        print(f"\tMemory usage_2 (grad - after collect): {memory_use}")
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self._loss_tracker_t1.update_state(losses_tracker['t1'])
        self._loss_tracker_t2.update_state(losses_tracker['t2'])
        self._loss_tracker_t3.update_state(losses_tracker['t3'])
        self.loss_tracker.update_state(loss_values)
        if self.__wandb:
            wandb.log(
                {
                    'lr': self._optimizer.learning_rate.numpy(),
                    'epochs': epoch,
                    'batch:': batch,
                    'strike_loss': self._loss_tracker_t1.result(),
                    'derivative_loss': self._loss_tracker_t2.result(),
                    'steps_error_loss': self._loss_tracker_t3.result(),
                    'overall_loss': self.loss_tracker.result(),
                    # Overall derivatives
                    'grads_magnitude': tf.reduce_mean(self._get_dv_dxi(self.N - 1)),
                    'analytical_grads': tf.reduce_mean(analytical_grads)
                }
            )
        return self.loss_tracker.result(), self._loss_tracker_t1.result(), self._loss_tracker_t2.result(), self._loss_tracker_t3.result()
   
    def reset_trackers(self):
        # Reset trackers
        self._loss_tracker_t1.reset_states()
        self._loss_tracker_t2.reset_states()
        self._loss_tracker_t3.reset_states()
        self.loss_tracker.reset_states()
        
    def plot_tracker_results(self, epoch: int):
        print(f'Epoch {epoch} Mean loss {self.loss_tracker.result()}')
        print(f'\tPartial losses:\n\t\tStrike loss:{self._loss_tracker_t1.result()}\n\t\tDerivative loss: {self._loss_tracker_t2.result()}\n\t\tSteps loss: {self._loss_tracker_t3.result()}')
        process = psutil.Process(os.getpid())
        memory_use = process.memory_info().rss / (1024 * 1024)
        print(f"\tMemory usage: {memory_use}")

    
    def get_losses_internal(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._loss_tracker_t1.result(), self._loss_tracker_t2.result(), self._loss_tracker_t3.result()
    
    def export_model_architecture(self, dot_img_file = 'model_architectures/each_step_at_a_time.png'):
        return tf.keras.utils.plot_model(self._custom_model, to_file=dot_img_file, show_shapes=True)

    def _get_dv_dx(self, features: tf.Tensor):
        samples = tf.shape(features)[0]
        batch_size = samples // self.N
        with tf.GradientTape() as tape:
            tape.watch(features)
            y = self._custom_model(features)
        grads = tape.gradient(y, features)
        if grads is not None:
            grads_reshaped = tf.reshape(grads[:, 0], (batch_size, self.N))
            grads_prediction = grads[:, 0]
            t_grads_reshaped = tf.reshape(grads[:, 1], (batch_size, self.N)) if grads.shape[1] > 1 else None
            t_grads_prediction = grads[:, 1] if grads.shape[1] > 1 else None
        else:
            grads_reshaped = tf.zeros((batch_size, self.N))
            grads_prediction = tf.zeros_like(features[:, 0])

            t_grads_reshaped = tf.zeros((batch_size, self.N))
            t_grads_prediction = tf.zeros_like(features[:, 0])
        self._grads = grads_reshaped
        self._grads_prediction = grads_prediction
        self._t_grads = t_grads_reshaped
        self._t_grads_prediction = t_grads_prediction
        return self._grads, self._grads_prediction, self._t_grads, self._t_grads_prediction

    
    def _get_dv_dxi(self, i, sample_idx = None):
        """_summary_

        Args:
            i (_type_): _description_
            sample_idx (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self._grads[:, i] if self._grads is not None else None

    def predict(self, 
        x:tf.Tensor, 
        delta_x:tf.Tensor,
        build_masks: bool = False,
        debug: bool = False
    ):
        pass
    
    def predict_loop(self,
        x:tf.Tensor,
        delta_x: tf.Tensor,
        build_masks: bool = False,
        debug: bool = False
    ):
        pass
        
    
    # Save model 
    def save_weights(self, path):
        self._custom_model.save_weights(path)
        
    # Save model 
    def load_weights(self, path):
        self._custom_model.load_weights(path)