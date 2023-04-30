# Imports
import tensorflow as tf
import numpy as np
# From
from tensorflow.keras import layers
from tensorflow import keras
# Utils
from utils.utils.utils import (
    FinanceUtils, 
    MathUtils
)

# TODO: Write better method definitions
class LGM_model(tf.keras.Model):

    def __init__(
        self,
        n_steps,
        T = 0,
        intermediate_dim=64,
        name="LGM_NN_model",
        verbose = False,
        *,
        sigma = None,
        batch_size = None,
        **kwargs
    ):
        """_summary_

        Args:
            n_steps (_type_): _description_
            intermediate_dim (int, optional): _description_. Defaults to 64.
            is_sequential (bool, optional): _description_. Defaults to False.
            name (str, optional): _description_. Defaults to "LGM_NN_model".
        """
        super(LGM_model, self).__init__(name=name, **kwargs)
        self.N = n_steps
        self._batch_size = batch_size
        # Local initializer
        initializer = tf.keras.initializers.GlorotUniform(seed = 6543210)
        # Tf model structure
        input_layer = keras.Input(shape=(n_steps, 2), name='input_nn')
        x = layers.GRU(intermediate_dim, 
                       kernel_initializer = initializer,
                       dropout = 0.2,
                       input_shape = (n_steps, 2),
                       return_sequences = True,
                       name = 'sequential_layer')(input_layer)
        output_layer = layers.Dense(units = 1, 
                                    activation = 'relu', 
                                    name = 'first_dense',
                                    kernel_initializer=initializer)(x)
        self._custom_model = keras.Model(
            inputs=[input_layer],
            outputs=[output_layer],
            name = name
        )
        # Metrics tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # Internal loss control
        self._loss_tracker_t1 = tf.keras.metrics.Mean(name="loss")
        self._loss_tracker_t2 = tf.keras.metrics.Mean(name="loss")
        self._loss_tracker_t3 = tf.keras.metrics.Mean(name="loss")
        # Internal loss results
        self._loss_tracker_t1_array = []
        self._loss_tracker_t2_array = []
        self._loss_tracker_t3_array = []
        # Duration each step
        self._deltaT = T / self.N
        # Constantes:
        self._ct = FinanceUtils.C(T, sigma_value = sigma)
        # Status variables
        self._grads, self._predictions = None, None
        # Verbose
        self._verbose = verbose
    
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
        with tf.GradientTape() as tape:
            x = tf.Variable(x, trainable = True)
            v = self.predict(x)
            predictions = tf.Variable(self._predictions, trainable = False)   
            loss = self._loss_lgm(x = x, v = v, predictions = predictions, N_steps = self.N)
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
    
    def custom_train_step(self, X, y = None, batch = 0, epoch = 0, start_time = None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            epoch (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        x = tf.constant(X)
        with tf.GradientTape() as tape:
            v, predictions = self.predict(x)
            predictions = tf.Variable(predictions)   
            loss = self.loss_lgm(x = X, 
                                 v = v, 
                                 predictions = predictions, 
                                 N_steps = self.N,
                                 verbose = self._verbose)
        grads = tape.gradient(loss, self.model.trainable_weights)
        # print(f'Grads: {len(grads)}')
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # Compute metrics
        self.loss_tracker.update_state(loss)
        if epoch % 100 == 0:
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
        return {'strike_loss': float(self._loss_tracker_t1.result()), 
                'derivative_strike_loss': float(self._loss_tracker_t2.result()), 
                'steps_error_loss': float(self._loss_tracker_t3.result())}
    
    def get_losses_internal_array(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {'strike_loss': self._loss_tracker_t1_array, 
                'derivative_strike_loss': self._loss_tracker_t2_array, 
                'steps_error_loss': self._loss_tracker_t3_array}

    def _get_dv_dx(self, X):
        """_summary_

        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """
        (x_dim, y_dim, z_dim) = X.shape
        self._grads = MathUtils.custom_diagonal_derivative(X, 
                                          self._custom_model)[:, :, 0]
        self._grads = tf.reshape(
            X,
            (
                x_dim,
                y_dim
            )
        )
        # Verbose to output
        if self._verbose:
            log_file = 'logs/20230217/grads_model.log'
            with open(log_file, 'a+') as f:
                f.write(f'Grads given X:\n')
                shape_x, shape_y, _ = features.shape
                for x_i in range(shape_x):
                    for y_i in range(shape_y):
                        f.write(f'{features[x_i, y_i]},')
                    f.write(f'\n')
        # Sanity purposes:
        # print(f'Grads shape: {self._grads.shape}')
        return self._grads
    
    def _get_dv_dxi(self, i, sample_idx = None):
        """_summary_

        Args:
            i (_type_): _description_
            sample_idx (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self._grads[:, i] if self._grads is not None else None
    
    # TODO: Separate from the model
    def loss_lgm(self, x: tf.Tensor, v: tf.Tensor, predictions: tf.Tensor, N_steps: np.int64, verbose: bool = False):
        """_summary_

        Args:
            x (tf.Tensor): _description_
            v (tf.Tensor): _description_
            predictions (tf.Tensor): _description_
            N_steps (np.int64): _description_

        Returns:
            _type_: _description_
        """
        betas = [0.02, 0.02, 1.00 * 10000]
        # Careful: Using global variable...
        len_path = N_steps
        # For f and f'
        xn_tensor = x[:, -1, 0]
        n_idx = int(len_path)
        # Loss given the strike function
        tn = np.float64(self._deltaT * len_path)
        function_at_strike = FinanceUtils.zero_bond_coupon(
            xn_tensor, 
            tn, 
            self._ct
        )
        strike_loss = tf.math.squared_difference(
                prediction[:, -1], 
                function_at_strike
        )
        # Autodiff f
        xn = tf.Variable(x[:, -1, 0], name = 'xn', trainable = True)
        tn = tf.Variable(np.float64(self._deltaT * len_path), name = 'tn', trainable=False)
        ct = tf.Variable(np.float64(self._ct), name = 'ct', trainable=False)
        with tf.GradientTape() as tape:
            y = FinanceUtils.zero_bond_coupon(xn, tn, ct)
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
        derivative_loss = tf.math.squared_difference(
            self._get_dv_dxi(n_idx - 1),
            df_dxn
        )
        # Epoch error per step
        error_per_step = tf.reduce_sum(
            tf.math.squared_difference(
                v[:, 1:], 
                predictions[:, 1:]), 
            axis = 1
        ) / N_steps
        # Record internal losses
        self._loss_tracker_t1.update_state(strike_loss)
        self._loss_tracker_t2.update_state(derivative_loss)
        self._loss_tracker_t3.update_state(error_per_step)
        # Weigth the errors
        strike_loss *= betas[0]
        derivative_loss *= betas[1]
        error_per_step *= betas[2]
        return tf.math.add(
            error_per_step, 
            tf.math.add(
                strike_loss, 
                derivative_loss
            )
        )

    def predict(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Steps
        samples, _, _ = X.shape
        # Swaping option at strike
        v = np.zeros((samples, self.N, 1), dtype = np.float64)
        predictions = self._custom_model(X)
        # Keep only the first value predicted
        v[:, 0, :] = predictions[:, 0, :]
        # Get the gradients
        grads = self._get_dv_dx(X)
        for i in range(0, self.N - 1):
            second_term = tf.math.multiply(
                tf.reshape(
                    grads[:, i],
                    (samples,
                     1)    
                ), 
                tf.reshape(
                    tf.math.subtract(
                        X[:, i + 1, 0], 
                        X[:, i, 0]
                    ),
                    (samples,
                     1)
                )
            )
            v[:, i + 1, :] = tf.math.add(
                predictions[:, i, :], 
                second_term
            )
        '''print(f'Grads: {grads} {grads.shape}')
        print(f'Predictions: {predictions}, {predictions.shape}')
        print(f'V: {v}, {v.shape}')'''
        v = tf.convert_to_tensor(v)
        return tf.reshape(v, (samples, self.N, 1)), predictions