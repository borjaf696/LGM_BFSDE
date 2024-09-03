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
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization
from tensorflow import keras

# Utils
from scripts.utils.utils.utils import FinanceUtils

# Loss functions
from scripts.losses.losses import Losses

# Wandb
import wandb


# TODO: Store all the metadata
class LgmSingleStep(tf.keras.Model):

    def __init__(
        self,
        n_steps,
        T=0,
        name="zerobond",
        *,
        dim=2,
        verbose=False,
        sigma=None,
        batch_size=None,
        phi=None,
        future_T=None,
        report_to_wandb=False,
        normalize=False,
        # First simulation
        data_sample=None,
        device = "cpu",
        **kwargs,
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
        self.N = tf.constant(n_steps, dtype=tf.float64)
        self.T = tf.constant(T, dtype=tf.float64)
        self.normalize = normalize
        self.device = device
        # Same for actives with out future
        self.future_T = tf.constant(future_T, dtype=tf.float64)
        self.batch_size = tf.constant(batch_size, dtype=tf.float64)
        self.expected_sample_size = self.N * self.batch_size
        # Phi function
        self.phi = phi
        # Type of active
        self.name_internal = name
        # Constantes:
        print(f"{'#'*100}")
        print(f"Strike time (T): {T}")
        print(f"Second strike time (TM): {self.future_T}")
        print(f"Number of steps per path: {n_steps}")
        print(f"Model name: {self.name_internal}")
        print(f"Batch size: {self.batch_size}")
        print(f"Expected sample size: {self.expected_sample_size}")
        print(f"{'#'*100}")
        # Model with time and value
        input_tmp = keras.Input(shape=(dim,), name=self.name_internal)
        if self.normalize:
            # Normalizer
            start_normalization_time = time.time()
            self.mean = tf.reduce_mean(data_sample, axis=0)
            self.var = tf.math.reduce_variance(data_sample, axis=0)
            self.epsilon = tf.constant(1e-7, dtype = tf.float64)
            """normalizer = Normalization(axis=-1)
            normalizer.adapt(
                data_sample
            )
            x = normalizer(
                input_tmp
            )"""
            print(f"[Normalization] Mean: {self.mean}")
            print(f"[Normalization] Var: {self.var}")
            print(f"[Normalization] Epsilon: {self.epsilon}")
            
            end_normalization_time = time.time()
            print(
                f"[Normalization] Normalization time: {end_normalization_time - start_normalization_time}s"
            )
        # Set first layer
        x = input_tmp
        # Configuration read from:
        # --- name
        # --- T, strike time
        configuration = None
        with open("scripts/configs/ff_config.json", "r+") as f:
            configuration = json.load(f)[name][str(T)]
        self.num_layers = configuration["layers"]
        print(f"Number of layers: {self.num_layers}")
        print(f'Number of hidden units: {configuration["hidden_units"]}')
        # Build dense layers
        self.dense_layers = []
        self.batch_norm_layers = []
        for i in range(self.num_layers):
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    units=configuration["hidden_units"],
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    name="internal_dense_" + str(i),
                )
            )
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        for i in range(0, self.num_layers):
            x = self.dense_layers[i](x)
            x = self.batch_norm_layers[i](x)
            x = tf.keras.layers.ReLU()(x)
        output_tmp = layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name="output_dense",
        )(x)
        self.custom_model = keras.Model(
            inputs=input_tmp, outputs=output_tmp, name=name
        )
        # Beta parameters
        self.beta1 = tf.Variable(1.0, trainable=False, dtype=tf.float64, name='beta1')
        self.beta2 = tf.Variable(1.0, trainable=False, dtype=tf.float64, name='beta2')
        self.beta3 = tf.Variable(1.0, trainable=False, dtype=tf.float64, name='beta3')
        self.beta4 = tf.Variable(1.0, trainable=False, dtype=tf.float64, name='beta4')
        # Set the loss function
        self.loss_lgm = Losses.adapted_loss_tf
        # Train Metrics tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # Train Internal management
        self.loss_tracker_t1 = tf.keras.metrics.Mean(name="strike_loss")
        self.loss_tracker_t2 = tf.keras.metrics.Mean(name="derivative_loss")
        self.loss_tracker_t3 = tf.keras.metrics.Mean(name="step_loss")
        self.loss_tracker_t4 = tf.keras.metrics.Mean(name="trivial_penalization")
        # Val Metrics tracker
        self.loss_tracker_val = tf.keras.metrics.Mean(name="total_loss")
        # Val Internal management
        self.loss_tracker_t1_val = tf.keras.metrics.Mean(name="strike_loss")
        self.loss_tracker_t2_val = tf.keras.metrics.Mean(name="derivative_loss")
        self.loss_tracker_t3_val = tf.keras.metrics.Mean(name="step_loss")
        self.loss_tracker_t4_val = tf.keras.metrics.Mean(name="trivial_penalization")
        # Duration each step
        self.deltaT = T / self.N
        # CT
        self.ct = FinanceUtils.C(T, sigma_value=sigma)
        # Status variables
        self.grads = None
        # Create masks
        self.__create_masks()
        # Tracked best model
        self.min_tracked_loss = 99999999
        self.best_weights = None
        # Verbose
        self.verbose = verbose
        # Track with wandb
        self.wandb = report_to_wandb
        
    def __create_masks_ori(self):
        sample_expected = np.int64(self.expected_sample_size)
        N = np.int64(self.N)
        idx_preds = np.array(range(0, sample_expected, N))
        mask_v = np.ones((sample_expected, 1))
        mask_v[idx_preds] = 0
        self._mask_v = tf.convert_to_tensor(
            (mask_v == 1), 
            dtype = tf.float64
        )
        self._mask_preds = tf.convert_to_tensor(
            (mask_v == 0),
            dtype = tf.float64
        )
        print(f'Positions to avoid from V {self.expected_sample_size - np.sum(self._mask_v)}')
        print(f'Positions to complete from V {np.sum(self._mask_preds)}')

        # Loss mask
        print(f'Loss mask: {self.expected_sample_size / self.N}')
        idx_preds = np.array(range(N - 1, sample_expected + 1, N))
        mask_loss = np.zeros((sample_expected, 1)) 
        mask_loss[idx_preds] = 1.0
        self._mask_loss = tf.reshape(
            tf.convert_to_tensor(
                (mask_loss == 1),
                dtype = tf.float64
            ),
            (sample_expected, 1)
        )
        print(f'Positions to avoid from loss {self.expected_sample_size - np.sum(self._mask_loss)}')
        print(f'Positions to complete from loss {np.sum(self._mask_loss)}')

    def __create_masks(self):
        idx_preds = tf.reshape(
            tf.range(0, self.expected_sample_size, self.N, dtype=tf.int64), (-1, 1)
        )
        mask_v = tf.ones((self.expected_sample_size, 1), dtype=tf.float64)
        mask_v = tf.tensor_scatter_nd_update(
            mask_v, idx_preds, tf.zeros_like(idx_preds, dtype=tf.float64)
        )
        self._mask_v = mask_v
        self._mask_preds = 1 - mask_v
        print(
            f"Positions to avoid from V {self.expected_sample_size - np.sum(self._mask_v)}"
        )
        print(f"Positions to complete from V {np.sum(self._mask_preds)}")

        idx_loss = tf.reshape(
            tf.range(self.N - 1, self.expected_sample_size, self.N, dtype=tf.int64),
            (-1, 1),
        )
        mask_loss = tf.zeros((self.expected_sample_size, 1), dtype=tf.float64)
        mask_loss = tf.tensor_scatter_nd_update(
            mask_loss, idx_loss, tf.ones_like(idx_loss, dtype=tf.float64)
        )

        self._mask_loss = mask_loss
        print(
            f"Positions to avoid from loss: {self.expected_sample_size - tf.reduce_sum(self._mask_loss).numpy()}"
        )
        print(
            f"Positions to complete from loss: {tf.reduce_sum(self._mask_loss).numpy()}"
        )
        number_of_elements = (
            tf.size(self._mask_loss).numpy() + tf.size(self._mask_preds).numpy()
        )
        element_size = tf.dtypes.as_dtype(self._mask_loss.dtype).size
        print(
            f"[MEMORY] Total memory consumed by masks: {number_of_elements * element_size / 2**30} Gb"
        )

    @property
    def model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.custom_model

    @property
    def metrics(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return [self.loss_tracker, self.mae_metric]

    def summary(self):
        """_summary_"""
        self.custom_model.summary()

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

    def define_compiler(self, optimizer="adam", learning_rate=1e-3):
        """_summary_

        Args:
            optimizer (str, optional): _description_. Defaults to 'adam'.
            learning_rate (_type_, optional): _description_. Defaults to 1e-3.
        """
        if optimizer == "adam":
            print(f"Optimizer set to {optimizer}")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def learning_rate(self, learning_rate):
        self.optimizer.learning_rate.assign(learning_rate)
        
    def fit_step(self, x: tf.Tensor, delta_x: tf.Tensor, apply_gradients: bool = True):
        return (
            self.custom_train_step_tf(x = x, delta_x = delta_x, apply_gradients = apply_gradients) 
            if self.device == "gpu" else 
            self.custom_train_step(x = x, delta_x = delta_x)
        )

    @tf.function(
        reduce_retracing=True, 
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.bool)  
        ]
    )
    def custom_train_step_tf(
        self, x: tf.Tensor, delta_x: tf.Tensor, apply_gradients: tf.bool = True
    ):
        with tf.GradientTape(persistent=False) as tape:
            v, predictions, grads = self.predict_loop_tf(x, delta_x=delta_x)
            v = tf.reshape(v, (self.batch_size, self.N))
            predictions = tf.reshape(predictions, (self.batch_size, self.N))
            loss_values, losses_tracker, _ = Losses.adapted_loss_tf(
                x=x,
                v=v,
                ct=self.ct,
                derivatives=grads[:, tf.cast(self.N - 1, tf.int32)],
                predictions=predictions,
                N_steps=self.N,
                T=self.T,
                TM=self.future_T,
                batch_size=self.batch_size,
                phi=self.phi,
                betas = [self.beta1, self.beta2, self.beta3]
            )
        if apply_gradients:
            trainable_vars = self.model.trainable_weights # + [self.beta1, self.beta2, self.beta3]
            grads = tape.gradient(loss_values, trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))
            self.loss_tracker.update_state(loss_values)
            self.loss_tracker_t1.update_state(losses_tracker["t1"])
            self.loss_tracker_t2.update_state(losses_tracker["t2"])
            self.loss_tracker_t3.update_state(losses_tracker["t3"])
            self.loss_tracker_t4.update_state(losses_tracker["t4"])
            return (
                self.loss_tracker.result(),
                self.loss_tracker_t1.result(),
                self.loss_tracker_t2.result(),
                self.loss_tracker_t3.result(),
                self.loss_tracker_t4.result()
            )
        self.loss_tracker_val.update_state(loss_values)
        self.loss_tracker_t1_val.update_state(losses_tracker["t1"])
        self.loss_tracker_t2_val.update_state(losses_tracker["t2"])
        self.loss_tracker_t3_val.update_state(losses_tracker["t3"])
        self.loss_tracker_t4_val.update_state(losses_tracker["t4"])
        return (
                self.loss_tracker_val.result(),
                self.loss_tracker_t1_val.result(),
                self.loss_tracker_t2_val.result(),
                self.loss_tracker_t3_val.result(),
                self.loss_tracker_t4_val.result()
            )

    def custom_train_step(self, x, batch = 1, epoch = 1, delta_x = None, apply_gradients = True):
        first_dim, _ = x.shape
        batch_size = np.int64(first_dim / self.N)
        #x = tf.constant(x)
        with tf.GradientTape() as tape:
            v, predictions, _ = self.predict(x, delta_x = delta_x)
            v = tf.reshape(v, (batch_size, self.N))
            predictions = tf.reshape(predictions, (batch_size, self.N))
            loss_values, losses_tracker, analytical_grads = Losses.loss_lgm(
                x = x, 
                v = v,
                ct = self.ct,
                derivatives = self._get_dv_dxi(tf.cast(self.N - 1, tf.int32)),  
                predictions = predictions, 
                N_steps = self.N,
                T = self.T,
                TM = self.future_T,
                phi = self.phi
            )
        grads = tape.gradient(loss_values, self.model.trainable_weights)
        if apply_gradients:
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.loss_tracker_t1.update_state(losses_tracker['t1'])
        self.loss_tracker_t2.update_state(losses_tracker['t2'])
        self.loss_tracker_t3.update_state(losses_tracker['t3'])

        self.loss_tracker.update_state(loss_values)
        if self.wandb:
            wandb.log(
                {
                    'lr': self.optimizer.learning_rate.numpy(),
                    'epochs': epoch,
                    'batch:': batch,
                    'strike_loss': self.loss_tracker_t1.result(),
                    'derivative_loss': self.loss_tracker_t2.result(),
                    'steps_error_loss': self.loss_tracker_t3.result(),
                    'overall_loss': self.loss_tracker.result(),
                    'grads_magnitude': tf.reduce_mean(self._get_dv_dxi(self.N - 1)),
                    'analytical_grads': tf.reduce_mean(analytical_grads)
                }
            )  
        return (
            float(self.loss_tracker.result()), 
            float(self.loss_tracker_t1.result()), 
            float(self.loss_tracker_t2.result()), 
            float(self.loss_tracker_t3.result())
        )

    def reset_trackers(self):
        # Reset trackers
        self.loss_tracker_t1.reset_state()
        self.loss_tracker_t2.reset_state()
        self.loss_tracker_t3.reset_state()
        self.loss_tracker.reset_state()

    def plot_tracker_results(self, epoch: int = 1, flag: str = "train"):
        tracked_loss = self.get_tracked_loss(flag = flag)
        print(f"Epoch {epoch} Mean loss {tracked_loss} Learning rate: {self.optimizer.learning_rate.numpy()}")
        if flag == "train":
            print(
                (
                    f"\tPartial losses during {flag}:"
                    f"\n\t\tStrike loss:{self.loss_tracker_t1.result()}"
                    f"\n\t\tDerivative loss: {self.loss_tracker_t2.result()}"
                    f"\n\t\tSteps loss: {self.loss_tracker_t3.result()}"
                    f"\n\t\tTrivial loss: {self.loss_tracker_t4.result()}"
                )
            )
        elif flag == "val":
            print(
                (
                    f"\tPartial losses during {flag}:"
                    f"\n\t\tStrike loss:{self.loss_tracker_t1_val.result()}"
                    f"\n\t\tDerivative loss: {self.loss_tracker_t2_val.result()}"
                    f"\n\t\tSteps loss: {self.loss_tracker_t3_val.result()}"
                    f"\n\t\tTrivial loss: {self.loss_tracker_t4_val.result()}"
                )
            )
            stored_new_model = self.evaluate_current_weights(tracked_loss = tracked_loss)
            print(f"\tStored new weights ({flag}): {stored_new_model}")
        process = psutil.Process(os.getpid())
        memory_use = process.memory_info().rss / (1024 * 1024)
        print(f"\tMemory usage: {memory_use}")
        
        
    def evaluate_current_weights(self, tracked_loss: float):
        if tracked_loss < self.min_tracked_loss:
            self.min_tracked_loss = tracked_loss
            self.best_weights = [np.copy(w.numpy()) for w in self.custom_model.weights]
            return True
        return False
    
    def set_best_weights(self):
        self.custom_model.set_weights(self.best_weights)
        
    def get_best_loss(self):
        return self.min_tracked_loss
        
    def get_tracked_loss(self, flag: str = "train"):
        return (self.loss_tracker.result() if flag == "train" else self.loss_tracker_val.result())

    def get_losses_internal(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return (
            self.loss_tracker_t1.result(),
            self.loss_tracker_t2.result(),
            self.loss_tracker_t3.result(),
        )

    def export_model_architecture(
        self, dot_img_file="model_architectures/each_step_at_a_time.png"
    ):
        return tf.keras.utils.plot_model(
            self.custom_model, to_file=dot_img_file, show_shapes=True
        )

    def _get_dv_dx_tf(self, x: tf.Tensor):
        with tf.GradientTape() as tape:
            tape.watch(x)
            if self.normalize:
                y = self.custom_model((x - self.mean) / (tf.sqrt(self.var + self.epsilon)))
            else:
                y = self.custom_model(x)
        grads = tape.gradient(y, {"xn": x})["xn"]
        samples = tf.cast(tf.cast(tf.shape(x)[0], tf.float64) / self.N, tf.float64)
        if grads is not None:
            grads_reshaped = tf.reshape(grads[:, 0], (samples, self.N))
            grads_prediction = grads[:, 0]
        else:
            grads_reshaped = tf.zeros((samples, self.N), dtype = tf.float64)
            grads_prediction = tf.zeros_like(x[:, 0], dtype = tf.float64)
        t_grads_reshaped = tf.zeros((samples, self.N), dtype = tf.float64)
        t_grads_prediction = tf.zeros_like(x[:, 0], dtype = tf.float64)
        self.grads = grads_reshaped
        return grads_reshaped, grads_prediction, t_grads_reshaped, t_grads_prediction
    
    def _get_dv_dx(self, features):
        samples, _ = features.shape
        batch_size = int(np.floor(samples / self.N))
        grads = []
        x_variable = tf.Variable(features, name = 'x')                
        with tf.GradientTape() as tape:
            tape.watch(x_variable)
            if self.normalize:
                x_variable_norm = (x_variable - self.mean) / tf.sqrt(self.var + self.epsilon)
            y = self.custom_model(
                x_variable_norm if self.normalize else x_variable
            )
        grads = tape.gradient(
            y, 
            {
                'x': x_variable
            }
        )
        self.grads = tf.reshape(grads['x'][:, 0], (batch_size, self.N))
        self.grads_prediction = grads['x'][:, 0]
        self.t_grads = tf.reshape(grads['x'][:, 1], (batch_size, self.N))
        self.t_grads_prediction = grads['x'][:, 1]
        return self.grads, self.grads_prediction, self.t_grads, self.t_grads_prediction

    def _get_dv_dxi(self, i):
        return self.grads[:, i] if self.grads is not None else None

    def predict_tf(
        self,
        x: tf.Tensor,
        delta_x: tf.Tensor,
        build_masks: bool = False,
        debug: bool = False,
    ):
        pass

    def predict(
        self,
        x: tf.Tensor,
        delta_x: tf.Tensor,
        build_masks: bool = False,
        debug: bool = False,
    ):
        pass

    def predict_loop(
        self,
        x: tf.Tensor,
        delta_x: tf.Tensor,
        build_masks: bool = False,
        debug: bool = False,
    ):
        pass

    # Save model
    def save_weights(self, path):
        self.custom_model.save_weights(path)

    # Save model
    def load_weights(self, path):
        self.custom_model.load_weights(path)
