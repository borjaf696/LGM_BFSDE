import tensorflow as tf
import numpy as np

class MathUtils():
    
    @staticmethod
    def custom_derivative_model(x, model, multioutput = True):
        h = 1e-3
        # Dimensions
        x_dim, y_dim, z_dim = x.shape
        # Gradient vector
        gradient = np.zeros((x_dim, y_dim, z_dim))
        for i in range(z_dim):
            # Vector for partial derivative estimation
            offset_tensor = np.zeros((x_dim, y_dim, 1))
            offset_tensor[:, i] = h
            offset_tensor = tf.convert_to_tensor(
                offset_tensor,
                dtype = tf.float64
            )
            # Constantes:
            denominator = h
            numerator = tf.math.subtract(
                model(
                    tf.math.add(x, offset_tensor)
                ), model(
                    tf.math.subtract(x, offset_tensor)
                )
            )
            denominator = 2 * h
            gradient[:, :, i] = tf.reshape(
                numerator / denominator,
                (x_dim, y_dim)
            )
        gradient = tf.convert_to_tensor(
            gradient,
            dtype = tf.float64
        )
        return gradient
    
    @staticmethod
    def custom_diagonal_derivative(x, model):
        h = 1e-1
        # Size x
        x_dim, y_dim, z_dim = x.shape
        # Gradient vector
        gradient = np.zeros((x_dim, y_dim, z_dim))
        for i in range(z_dim):
            for j in range(y_dim):
                # Vector for partial derivative estimation
                offset_tensor = np.zeros((x_dim, y_dim, z_dim))
                offset_tensor[:, j, i] = h
                offset_tensor = tf.convert_to_tensor(offset_tensor,
                                                    dtype = tf.float64)
                # Constantes:
                denominator = h
                numerator = tf.math.subtract(
                    model(
                        tf.math.add(x, offset_tensor)
                    ), model(
                        tf.math.subtract(x, offset_tensor)
                    )
                )
                denominator = 2 * h
                gradient[:, j, i] = numerator[:, j, 0] / denominator
        gradient = tf.convert_to_tensor(gradient,
                                            dtype = tf.float64)
        return gradient
class MLUtils():
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin
    ## TODO:
    # ------ Adapt fit to work only with training and then transform given the
    # ------ the previous transformation
    class NormalizeColumns(BaseEstimator, TransformerMixin):
        """ S-normalization

        Args:
            BaseEstimator (_type_): _description_
            TransformerMixin (_type_): _description_
        """
        def __init__(self, columns=None):
            self.columns = columns
            self.__sufix = '_transformed'
        
        def fit(self, X, y=None):
            self.__mean = np.mean(X, axis = 0)
            self.__std = np.std(X, axis = 0)
            return self

        def transform(self, X, y=None):
            cols_to_transform = list(X.columns)

            if self.columns:
                cols_to_transform = self.columns
            cols_renamed = []
            for column in cols_to_transform:
                cols_renamed.append(
                    column + self.__sufix
                )
            
            X[cols_renamed] = (X[cols_to_transform] - self.__mean) / (self.__std)
            return X
    
    @staticmethod
    def get_pipeline(transformations = None):
        from sklearn.pipeline import Pipeline
        transformation_map = {
            'normalization': MLUtils.NormalizeColumns()
        }
        pipeline_steps = []
        for transformation in transformations:
            pipeline_steps.append(
                (transformation, transformation_map[transformation])
            )
        pipe = Pipeline(steps = pipeline_steps)
        return pipe
            
class FinanceUtils():
    @staticmethod
    def sigma(t, sigma_0 = 0.0075):
        """_summary_

        Args:
            t (_type_): _description_
            sigma_0 (float, optional): _description_. Defaults to 0.0075.

        Returns:
            _type_: _description_
        """
        return sigma_0 + (t // 2) * 0.0005
    
    @staticmethod
    def C(t, sigma_value = 0.0075):
        """_summary_

        Args:
            t (_type_): _description_
            sigma (_type_, optional): _description_. Defaults to sigma.

        Returns:
            _type_: _description_
        """
        import scipy.integrate as integrate
        return integrate.quad(lambda x: FinanceUtils.sigma(t = x, 
                                              sigma_0 = sigma_value)**2, 0, t)[0]      
    
class ZeroBond():  
    @staticmethod
    def H_tensor(t, kappa = 2):
        """_summary_

        Args:
            t (_type_): _description_
            kappa (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        return (1 - tf.math.exp(-kappa * t)) / kappa  
    @staticmethod
    def D_tensor(t, r = 0.03):
        """_summary_

        Args:
            t (_type_): _description_
            r (float, optional): _description_. Defaults to 0.03.

        Returns:
            _type_: _description_
        """
        return tf.math.exp(-r * t)
    @staticmethod
    def N_tensor(t, xt, ct):
        """_summary_

        Args:
            t (_type_): _description_
            xt (_type_): _description_
            ct (_type_): _description_

        Returns:
            _type_: _description_
        """
        return tf.math.multiply(1/ZeroBond.D_tensor(t), tf.math.exp(tf.math.add(tf.math.multiply(ZeroBond.H_tensor(t), xt), 0.5 * tf.math.square(ZeroBond.H_tensor(t)) * ct)))
    @staticmethod
    def Z_tensor(xt, t, T, ct = None):
        """_summary_

        Args:
            xt (_type_): _description_
            t (_type_): _description_
            T (_type_): _description_
            ct (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert ct is not None
        return ZeroBond.D_tensor(T) * tf.math.exp(-0.5 * ZeroBond.H_tensor(T)**2 * ct - ZeroBond.H_tensor(T)*xt) * ZeroBond.N_tensor(t, xt, ct)
    @staticmethod
    def D(t, r = 0.03):
        """_summary_

        Args:
            t (_type_): _description_
            r (float, optional): _description_. Defaults to 0.03.

        Returns:
            _type_: _description_
        """
        return np.exp(-r * t)
    @staticmethod
    def H(t, kappa = 2):
        """_summary_

        Args:
            t (_type_): _description_
            kappa (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        return (1 - np.exp(-kappa * t)) / kappa
    @staticmethod
    def N(t, xt, ct):
        """_summary_

        Args:
            t (_type_): _description_
            xt (_type_): _description_
            ct (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1/ZeroBond.D(t) * np.exp(ZeroBond.H(t) * xt + 0.5 * ZeroBond.H(t)**2 * ct)
    @staticmethod
    def Z(xt, t, T, ct = None):
        """_summary_

        Args:
            xt (_type_): _description_
            t (_type_): _description_
            T (_type_): _description_
            ct (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert ct is not None
        return ZeroBond.D(T) * np.exp(-0.5 * ZeroBond.H(T)**2 * ct - ZeroBond.H(T)*xt) * ZeroBond.N(t, xt, ct)
    @staticmethod
    def exponent(xt, t, T, ct = None):
        """_summary_

        Args:
            xt (_type_): _description_
            t (_type_): _description_
            T (_type_): _description_
            ct (_type_, optional): _description_. Defaults to None.
        """
        assert ct is not None
        return np.exp(-0.5 * ZeroBond.H(T) ** 2 * ct - ZeroBond.H(T) * xt)
    
    @staticmethod
    @tf.function
    def Z_normalized(xn, tn, T, ct):
        """_summary_
        Args:
            xn (_type_): _description_
            n (_type_): _description_
            ct (_type_): _description_
        Returns:
            _type_: _description_
        """
        return tf.math.multiply(ZeroBond.Z_tensor(xn, tn, T, ct), 1/ ZeroBond.N_tensor(tn, xn, ct))

# Constants
TAUS = {
    3: 0.25,
    6: 0.5,
    12: 1.0
}
TIMES = {
    3: 0.25,
    6: 0.5,
    12:1.0
}

class IRS():
    @staticmethod
    @tf.function
    def IRS(xn, 
            T,
            TN, 
            ct, 
            period = 6,
            K = 0.03):
        tau = TAUS[period]
        time_add = TIMES[period]
        # Internal parameter
        num_times = np.float64((TN - T) / time_add)
        variable = (1 - ZeroBond.Z_tensor(xn, T, TN, ct))
        fixed = 0.
        for i in range(1.0, num_times + 1):
            fixed += ZeroBond.Z(xn, T, T + i * time_add, ct)
        fixed *= tau * K
        return variable + fixed
        
    @staticmethod
    @tf.function
    def IRS_normalized(xn, 
            T,
            TN, 
            ct, 
            period = 6,
            K = 0.03):
        tau = TAUS[period]
        time_add = TIMES[period]
        # Internal parameter
        first_val = np.float64(1.0)
        num_times = (TN - T) / time_add
        variable = (1 - ZeroBond.Z_tensor(xn, T, TN, ct))
        fixed = tf.zeros(
            tf.shape(xn),
            dtype = tf.float64
        )
        for i in range(first_val, num_times + 1):
            fixed += ZeroBond.Z_tensor(xn, T, T + i * time_add, ct)
        fixed *= tau * K
        # Normalize factor
        N = ZeroBond.N_tensor(T, xn, ct)
        return variable + fixed / N
class Swaption():
    
    @staticmethod
    def anuality_swap(
        xn, 
        T,
        TN, 
        ct, 
        period = 6,  
    ):
        # Anuality params
        tau = TAUS[period]
        time_add = TIMES[period]
        num_times = int(np.float64((T - TN) / time_add))
        fixed = np.zeros(
            np.shape(xn)
        )
        for i in range(1, num_times + 1):
            fixed += ZeroBond.Z(xn, T, TN + i * time_add, ct)
        
        return tau * fixed
    @staticmethod
    def par_swap(
        xn,
        t,
        Ti,
        Tm,
        ct,
        period = 6,
    ):  
        pi = ZeroBond.Z(xn, t, Ti, ct)
        pm = ZeroBond.Z(xn, t, Tm, ct)
        fixed = Swaption.anuality_swap(
            xn,
            Ti,
            Tm,
            ct,
            period
        )
            
        return (pi - pm) / fixed
    
    @staticmethod
    def positive_part_parswap(
        xn,
        t,
        Ti,
        Tm,
        ct,
        period = 6,
        K = 0.03
    ):
        def max_between_zero(x):
            return np.maximum(x, 0) 
        par_swap = Swaption.par_swap(
            xn,
            t,
            Ti,
            Tm,
            ct,
            period
        )
        
        return np.apply_along_axis(max_between_zero, axis = 1, arr = par_swap - K)
    
    @staticmethod
    def density_normal(
        xT,
        xn,
        t,
        Ti,
        Tm
    ):
        mu = xn
        std = FinanceUtils.C_tensor(xn, t, Ti) * FinanceUtils.C_tensor(xn, Ti, Tm)
        
        return 1 / np.sqrt(2 * np.pi * std**2) * np.exp((xT-mu)**2 / (2 * std**2))
    
    @staticmethod
    def Swaption_test(
        xn,
        t,
        Ti,
        Tm,
        ct,
        period = 6,
        K = 0.03
    ):
        def integra_swap(xT):
            par_swap = Swaption.positive_part_parswap(
                xT,
                t,
                Ti,
                Tm,
                ct,
                period,
                K
            )
            density_normal = Swaption.density_normal(
                xT,
                xn,
                t,
                Ti,
                Tm
            )
            anuality_term = Swaption.anuality_swap(
                xT,
                t,
                Ti,
                Tm,
                ct,
                period
            )
            return par_swap * anuality_term * density_normal
            
        import scipy.integrate as integrate
        return integrate.quad(lambda x: integra_swap(x), -np.inf, np.inf)[0]     
    
    @staticmethod
    @tf.function
    def Swaption(xn, 
            T,
            TN, 
            ct, 
            period = 6,
            K = 0.03):
        tau = TAUS[period]
        time_add = TIMES[period]
        # Internal parameter
        first_val = np.float64(1.0)
        num_times = (TN - T) / time_add
        variable = (1 - ZeroBond.Z_tensor(xn, T, TN, ct))
        fixed = tf.zeros(
            tf.shape(xn),
            dtype = tf.float64
        )
        for i in range(first_val, num_times + 1):
            fixed += ZeroBond.Z_tensor(xn, T, T + i * time_add, ct)
        fixed *= tau * K
        # Fix shape for swaption
        tensor_irs = tf.reshape(
            (variable + fixed),
            (variable.shape[0], 1)
        )
        zero_mask = tf.zeros(
            (tensor_irs.shape[0], 1), 
            dtype = tensor_irs.dtype
        )
        tensor_irs = tf.concat(
            [
                tensor_irs,
                zero_mask
            ],
            axis = 1
        )
        return tf.reduce_max(tensor_irs, axis = 1)
        
    @staticmethod
    @tf.function
    def Swaption_normalized(xn, 
            T,
            TN, 
            ct, 
            period = 6,
            K = 0.03):
        tau = TAUS[period]
        time_add = TIMES[period]
        # Internal parameter
        first_val = np.float64(1.0)
        num_times = (TN - T) / time_add
        variable = (1 - ZeroBond.Z_tensor(xn, T, TN, ct))
        fixed = tf.zeros(
            tf.shape(xn),
            dtype = tf.float64
        )
        for i in range(first_val, num_times + 1):
            fixed += ZeroBond.Z_tensor(xn, T, T + i * time_add, ct)
        fixed *= tau * K
        # Normalize factor
        N = ZeroBond.N_tensor(T, xn, ct)
        # Fix shape for swaption
        tensor_irs = tf.reshape(
            (variable + fixed),
            (variable.shape[0], 1)
        )
        zero_mask = tf.zeros(
            (tensor_irs.shape[0], 1), 
            dtype = tensor_irs.dtype
        )
        tensor_irs = tf.concat(
            [
                tensor_irs,
                zero_mask
            ],
            axis = 1
        )
        return tf.reduce_max(tensor_irs, axis = 1) / N