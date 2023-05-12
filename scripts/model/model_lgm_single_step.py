# Load configuration
import json
# Imports
import tensorflow as tf
import numpy as np
# From
from tensorflow.keras import layers
from tensorflow import keras
# Utils
from utils.utils.utils import FinanceUtils
# Losses
from losses.losses import Losses
# TODO: Write better method definitions
class LGM_model_one_step(tf.keras.Model):

    def __init__(
        self,
        n_steps,
        T = 0,
        name="zerobond",
        *,
        verbose = False,
        sigma = None,
        batch_size = None,
        phi = None,
        future_T = None,
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
        super(LGM_model_one_step, self).__init__(name=name, **kwargs)
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
        # Model with time and value
        input_tmp = keras.Input(shape = (2, ), 
                                name = 'input_nn')
        initializer = tf.keras.initializers.GlorotUniform(seed = 6543210)
        # Configuration read from:
        # --- name
        # --- T, strike time
        configuration = None
        with open('scripts/configs/ff_config.json', 'r+') as f:
            configuration = json.load(f)[name][str(T)]
        self.__num_layers = configuration['layers']
        print(f'Number of layers: {self.__num_layers}')
        print(f'Number of hidden units: {configuration["hidden_units"]}')
        for i in range(self.__num_layers):
            objective_layer = x if i >0 else input_tmp
            x = layers.Dense(units = configuration['hidden_units'],
                    activation = 'relu',
                    kernel_initializer = initializer,
                    name = 'internal_relu_dense_'+str(i))(objective_layer)
        output_tmp = layers.Dense(units = 1, 
                                    activation = 'relu', 
                                    kernel_initializer = initializer,
                                    name = 'output_relu_dense')(x)
        self._custom_model = keras.Model(
            inputs = input_tmp,
            outputs = output_tmp,
            name = name)
        # Metrics tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # Internal management
        self._loss_tracker_t1 = tf.keras.metrics.Mean(name="strike_loss")
        self._loss_tracker_t2 = tf.keras.metrics.Mean(name="derivative_loss")
        self._loss_tracker_t3 = tf.keras.metrics.Mean(name="step_loss")
        # Internal loss results
        self._loss_tracker_t1_array = []
        self._loss_tracker_t2_array = []
        self._loss_tracker_t3_array = []
        # Duration each step
        self._deltaT = T / self.N
        # Constantes:
        print(f'T: {T}')
        self._ct = FinanceUtils.C(T, sigma_value = sigma)
        # Status variables
        self._grads, self._predictions = None, None
        # Create masks
        self.__create_masks()
        # Verbose
        self._verbose = verbose
        
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
        idx_preds = np.array(range(0, self._expected_sample_size, self.N))[1:] - 1
        mask_loss = np.ones((self._expected_sample_size, 1)) 
        mask_loss[idx_preds] = 1.0
        self._mask_loss = tf.reshape(
            tf.convert_to_tensor(
                (mask_loss == 1),
                dtype = tf.float64
            ),
            (self._expected_sample_size, 1)
        )
    
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
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, y = data
        x = tf.Variable(x, trainable = True)
        with tf.GradientTape() as tape:
            tape.watch(x)
            v = self.predict(x)
            predictions = tf.Variable(self._predictions, trainable = False)   
            loss = self._loss_lgm(x = x, v = v, predictions = predictions, N_steps = N_steps)
        # Get trainable vars
        trainable_vars = self.trainable_weights
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        # Compute metrics
        self.loss_tracker.update_state(loss)
        # Valor erroneo
        self.mae_metric.update_state(v, y)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}
    
    def define_compiler(self, optimizer = 'adam', learning_rate = 1e-3):
        """_summary_

        Args:
            optimizer (str, optional): _description_. Defaults to 'adam'.
            learning_rate (_type_, optional): _description_. Defaults to 1e-3.
        """
        if optimizer == 'adam':
            print(f'Optimizer set to {optimizer}')
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    def custom_train_step(self, X, y = None, batch = 0, epoch = 0, start_time = None, delta_x = None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            epoch (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        first_dim, _ = X.shape
        batch_size = np.int64(first_dim / self.N)
        x = tf.constant(X)
        with tf.GradientTape() as tape:
            v, predictions = self.predict(x, delta_x = delta_x)
            v = tf.reshape(v, (batch_size, self.N))
            predictions = tf.reshape(predictions, (batch_size, self.N))
            loss, losses_tracker = Losses.loss_lgm(x = x, 
                                 v = v,
                                 ct = self._ct,
                                 derivatives = self._get_dv_dxi(self.N - 1),  
                                 predictions = predictions, 
                                 N_steps = self.N,
                                 verbose = self._verbose,
                                 T = self.T,
                                 TN = self.future_T,
                                 phi = self.__phi,
                                 mask_loss = self._mask_loss)
        grads = tape.gradient(loss, self.model.trainable_weights)
        # print(f'Grads: {len(grads)}')
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # Losses tracker
        self._loss_tracker_t1.update_state(losses_tracker['t1'])
        self._loss_tracker_t2.update_state(losses_tracker['t2'])
        self._loss_tracker_t3.update_state(losses_tracker['t3'])
        # Compute metrics
        self.loss_tracker.update_state(loss)
        if epoch % 10 == 0:
            if start_time is None:
                print(f'Epoch {epoch} Mean loss {self.loss_tracker.result()}')
            else:
                import time
                print(f'Epoch {epoch} Batch {batch} Mean loss {self.loss_tracker.result()} Time per epoch: {(time.time() - start_time) / (epoch + 1)}s')
                print(f'\tPartial losses:\n\t\tStrike loss:{self._loss_tracker_t1.result()}\n\t\tDerivative loss: {self._loss_tracker_t2.result()}\n\t\tSteps loss: {self._loss_tracker_t3.result()}')
                self._loss_tracker_t1_array.append(self._loss_tracker_t1.result())
                self._loss_tracker_t2_array.append(self._loss_tracker_t2.result())
                self._loss_tracker_t3_array.append(self._loss_tracker_t3.result())         
        # Store losses
        return float(self.loss_tracker.result()), float(self._loss_tracker_t1.result()), float(self._loss_tracker_t2.result()), float(self._loss_tracker_t3.result())
    
    def get_losses_internal(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._loss_tracker_t1.result(), self._loss_tracker_t2.result(), self._loss_tracker_t3.result()
    
    def get_losses_internal_array(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {'strike_loss': self._loss_tracker_t1_array, 
                'derivative_strike_loss': self._loss_tracker_t2_array, 
                'steps_error_loss': self._loss_tracker_t3_array}
    
    def export_model_architecture(self, dot_img_file = 'model_architectures/each_step_at_a_time.png'):
        return tf.keras.utils.plot_model(self._custom_model, to_file=dot_img_file, show_shapes=True)

    def _get_dv_dx(self, features):
        """_summary_

        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """
        samples, _ = features.shape
        batch_size = int(np.floor(samples / self.N))
        grads = []
        x_variable = tf.Variable(features, name = 'x')                
        with tf.GradientTape() as tape:
            tape.watch(x_variable)
            y = self._custom_model(x_variable)
        # This represents dV/dX
        grads = tape.gradient(y, {
            'x':x_variable
        })
        # Take only X partial derivatives
        self._grads = tf.reshape(grads['x'][:, 0], (batch_size, self.N))
        self._grads_prediction = grads['x'][:, 0]
        # Verbose to output
        if self._verbose:
            log_file = 'logs/20230217/grads_model.log'
            with open(log_file, 'a+') as f:
                f.write(f'Grads given X:\n')
                shape_x, shape_y = features.shape
                for x_i in range(shape_x):
                    for y_i in range(shape_y):
                        f.write(f'{features[x_i, y_i]},')
                    f.write(f'\n')
        # Sanity purposes:
        # print(f'Grads shape: {self._grads.shape}')
        return self._grads, self._grads_prediction
    
    def _get_dv_dxi(self, i, sample_idx = None):
        """_summary_

        Args:
            i (_type_): _description_
            sample_idx (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self._grads[:, i] if self._grads is not None else None

    def predict(self, X:tf.Tensor, 
                delta_x:tf.Tensor,
                build_masks: bool = False):
        """_summary_

        Args:
            X (tf.Tensor): _description_

        Returns:
            _type_: _description_
        """
        sample_size = X.shape[0]
        batch_size = X.shape[0] // self.N
        # Predict
        predictions = self._custom_model(X)
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
        '''print(f'V: {v[0]}\nPredictions: {predictions[0]}')
        print(f'V: {v[1]}\nPredictions: {predictions[1]}')
        print(f'Grads: {grads[1]}\n')
        sys.exit()'''
        return v, predictions
    
    # Save model 
    def save_weights(self, path):
        self._custom_model.save_weights(path)
        
    # Save model 
    def load_weights(self, path):
        self._custom_model.load_weights(path)