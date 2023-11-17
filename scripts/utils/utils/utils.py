import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from scipy.stats import norm
from typing import (
    Any, 
    List,
    Dict
)

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

# Cyclic learning rate with decay
class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, max_lr, step_size, decay=1., mode='triangular'):
        super().__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.decay = decay
        self.mode = mode

        self.clr_iterations = 0.

    def __call__(self, step):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)

        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        else:
            raise NotImplementedError("mode '{}' is not implemented".format(self.mode))

        self.clr_iterations += 1
        # Apply decay
        return lr * (self.decay ** (self.clr_iterations / self.step_size))
    
class TFUtils():
    @staticmethod
    def custom_reshape(tensor):
        if tf.rank(tensor) == 1:
            return tf.reshape(tensor, [-1, 1])
        return tensor 

    @staticmethod
    def safe_divide(numerator, denominator, value_if_zero=0.0):
        denominator_safe = tf.where(denominator == 0, tf.ones_like(denominator), denominator)
        result = numerator / denominator_safe
        result_safe = tf.where(denominator == 0, tf.ones_like(result) * value_if_zero, result)
        return result_safe
    
class Utils():
    
    @staticmethod
    def get_features_with_pattern(df: pd.DataFrame, pattern: str, extra_cols: List[str]):
        features = []
        for column in df.columns.values:
            if column[0] == pattern:
                features.append(
                    column
                )
        features += extra_cols
        
        return features
    
    @staticmethod
    def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()
    
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
    def central_difference_gradients(x, model, h=1e-3):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        gradients = np.zeros_like(x)
        for j in range(x.shape[1]):
            h_tensor = np.zeros_like(x)
            h_tensor[:, j] = h
            f_x_plus_h = model(x + h_tensor)
            f_x_minus_h = model(x - h_tensor)
            gradient = (f_x_plus_h - f_x_minus_h) / (2 * h)
            gradients[:, j] = np.squeeze(gradient)
            
        gradients = tf.convert_to_tensor(
            gradients,
            dtype = tf.float32
        )

        return gradients
    
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
        
    @tf.function
    def sin_activation(x):
        return tf.math.sin(x)
    
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
    def H_tensor(t, kappa = 0.02):
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
    def H(t, kappa = 0.02):
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
    def Z_norm(xn, tn, T, ct = None):
        """_summary_
        Args:
            xn (_type_): _description_
            n (_type_): _description_
            ct (_type_): _description_
        Returns:
            _type_: _description_
        """
        assert ct is not None
        return  ZeroBond.Z(xn, tn, T, ct)/ZeroBond.N(tn, xn, ct)
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
    # @tf.function
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
    def IRS_normalized_np(
        xn, 
        T,
        TN, 
        ct, 
        period = 6,
        K = 0.03
    ):
        tau = TAUS[period]
        time_add = TIMES[period]
        num_times = np.float64((TN - T) / time_add)
        variable = (1 - ZeroBond.Z(xn, T, TN, ct))
        fixed = 0.
        for i in range(1.0, num_times + 1):
            fixed += ZeroBond.Z(xn, T, T + i * time_add, ct)
        fixed *= -tau * K
        return variable + fixed
    
    @staticmethod
    def IRS_normalized(xn, 
            T,
            TN, 
            ct, 
            period = 6,
            K = 0.03):
        tau = TAUS[period]
        time_add = TIMES[period]
        # Internal parameter
        first_val = int(1.0)
        num_times = int((TN - T) / time_add)
        variable = (1 - ZeroBond.Z_tensor(xn, T, TN, ct))
        fixed = tf.zeros(
            tf.shape(xn),
            dtype = tf.float64
        )
        for i in range(first_val, num_times + 1):
            fixed += ZeroBond.Z_tensor(xn, T, T + i * time_add, ct)
        fixed *= - tau * K
        # Normalize factor
        N = ZeroBond.N_tensor(T, xn, ct)
        return (variable + fixed) / N
    
    # TODO: Adapt
    @staticmethod
    def IRS_test(
        xn,
        t,
        Ti,
        Tm,
        ct,
        period = 6,
        K = 0.03,
        sigma_value = 0.01,
        predictions = None,
        debug = False
    ):
        def integra_swap(xT, xnj, ct, cT):
            par_swap = Swaption.positive_part_parswap(
                xn = xT,
                Ti = Ti,
                Tm = Tm,
                ct = cT,
                period = period,
                K = K
            )
            density_normal = Swaption.density_normal(
                xT,
                xnj,
                ct = ct,
                cT = cT
            )
            anuality_term = Swaption.anuality_swap(
                xT,
                Ti,
                Tm,
                cT,
                period
            )
            return (par_swap * anuality_term * density_normal) / ZeroBond.N(Ti, xT, cT)
            
        import scipy.integrate as integrate
        ct_dict = dict()
        ts_unique = np.unique(t)
        for t_unique in ts_unique:
            ct_dict[t_unique] = FinanceUtils.C(
                t_unique,
                sigma_value
            )
        cT = FinanceUtils.C(
            Ti,
            sigma_value
        )    
        swaption_results = []
        # TODO: Clean
        if debug == True:
            i = (t == 0)
            prediction = np.float64(predictions[i][0])
            xni = np.float64(xn[i][0])
            ct = np.float64(ct[i][0])
            integrate_swap = integrate.fixed_quad(
                        integra_swap, 
                        xni - 6 * np.sqrt((cT - ct)), 
                        xni + 6 * np.sqrt((cT - ct)), 
                        n = 1000,
                        args = (
                            xni,
                            ct,
                            cT    
                        )
                    ) 
            xni = tf.constant([xni], dtype = tf.float64)
            Ti = np.float64(Ti)
            Tm = np.float64(Tm)
            v_zero = Swaption.Swaption_at_zero(
                cT,
                Ti,
                Tm,
                period = period,
                K = K
            )
            print(f'Prediction: {prediction}')
            print(f'Integrate swap: {integrate_swap}')
            print(f'Analytical result: {v_zero}')
            sys.exit(0)
        for i, _ in enumerate(t):
            xni = xn[i]
            cti = ct[i]
            swaption_results.append(
                integrate.fixed_quad(
                    integra_swap, 
                    xni - 6 * np.sqrt((cT - cti)), 
                    xni + 6 * np.sqrt((cT - cti)), 
                    args = (
                        xni,
                        cti,
                        cT    
                    ))[0]    
            )
        return swaption_results  
    
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
        num_times = int(np.float64((TN - T) / time_add))
        fixed = 0
        for i in range(1, num_times + 1):
            fixed += ZeroBond.Z(xn, T, T + i * time_add, ct)
        return tau * fixed
    
    @staticmethod
    def par_swap(
        xn,
        Ti,
        Tm,
        ct,
        period = 6,
    ):  
        pi = ZeroBond.Z(xn, Ti, Ti, ct)
        pm = ZeroBond.Z(xn, Ti, Tm, ct)
        # For anuality we only require last Zeta_t
        fixed = Swaption.anuality_swap(
            xn,
            Ti,
            Tm,
            ct,
            period
        )
        return (pi - pm) / (fixed)
    
    @staticmethod
    def positive_part_parswap(
        xn,
        Ti,
        Tm,
        ct,
        period = 6,
        K = 0.03
    ):
        par_swap = Swaption.par_swap(
            xn,
            Ti,
            Tm,
            ct,
            period
        )
        return np.maximum(0, par_swap - K)
    
    @staticmethod
    def density_normal(
        xT,
        xn,
        ct,
        cT
    ):
        mu = xn
        std = np.sqrt(cT - ct)
        p = norm.pdf(xT, mu, std)
        return p
    
    @staticmethod
    def Swaption_test(
        xn,
        t,
        Ti,
        Tm,
        ct,
        period = 6,
        K = 0.03,
        sigma_value = 0.01,
        predictions = None,
        debug = False
    ):
        def integra_swap(xT, xnj, ct, cT):
            par_swap = Swaption.positive_part_parswap(
                xn = xT,
                Ti = Ti,
                Tm = Tm,
                ct = cT,
                period = period,
                K = K
            )
            density_normal = Swaption.density_normal(
                xT,
                xnj,
                ct = ct,
                cT = cT
            )
            anuality_term = Swaption.anuality_swap(
                xT,
                Ti,
                Tm,
                cT,
                period
            )
            return (par_swap * anuality_term * density_normal) / ZeroBond.N(Ti, xT, cT)
            
        import scipy.integrate as integrate
        ct_dict = dict()
        ts_unique = np.unique(t)
        for t_unique in ts_unique:
            ct_dict[t_unique] = FinanceUtils.C(
                t_unique,
                sigma_value
            )
        cT = FinanceUtils.C(
            Ti,
            sigma_value
        )    
        swaption_results = []
        # TODO: Clean
        if debug == True:
            i = (t == 0)
            prediction = np.float64(predictions[i][0])
            xni = np.float64(xn[i][0])
            ct = np.float64(ct[i][0])
            integrate_swap = integrate.fixed_quad(
                        integra_swap, 
                        xni - 6 * np.sqrt((cT - ct)), 
                        xni + 6 * np.sqrt((cT - ct)), 
                        n = 1000,
                        args = (
                            xni,
                            ct,
                            cT    
                        )
                    ) 
            xni = tf.constant([xni], dtype = tf.float64)
            Ti = np.float64(Ti)
            Tm = np.float64(Tm)
            v_zero = Swaption.Swaption_at_zero(
                cT,
                Ti,
                Tm,
                period = period,
                K = K
            )
            print(f'Prediction: {prediction}')
            print(f'Integrate swap: {integrate_swap}')
            print(f'Analytical result: {v_zero}')
            sys.exit(0)
        for i, _ in enumerate(t):
            xni = xn[i]
            cti = ct[i]
            swaption_results.append(
                integrate.fixed_quad(
                    integra_swap, 
                    xni - 6 * np.sqrt((cT - cti)), 
                    xni + 6 * np.sqrt((cT - cti)), 
                    args = (
                        xni,
                        cti,
                        cT    
                    ))[0]    
            )
        return swaption_results  
    
    @staticmethod
    def Swaption_at_zero(
        ceta_T,
        T,
        TN,
        period = 6,
        K = 0.03,
    ):    
        from scipy.optimize import fsolve
        import numpy as np

        def fystar(x):
            # First term
            exponential_first_term = - deltaHN * x
            exponential_second_term = - 0.5 * deltaHN**2*ceta_T
            first_term = ZeroBond.D(TN) * np.exp(exponential_first_term + exponential_second_term)
            # Second term
            second_term = 0
            for i in range(1, num_times + 1):
                Ti = T + i * time_add
                deltaHi = ZeroBond.H(Ti) - ZeroBond.H(T)
                exponential_first_term_second = -deltaHi * x
                exponential_second_term_second = -0.5 * deltaHi**2*ceta_T
                second_term += ZeroBond.D(Ti) * np.exp(exponential_first_term_second + exponential_second_term_second)
            second_term *= tau * K
            # Third term
            third_term = - ZeroBond.D(T)
            return first_term + second_term + third_term
        
        print(f'Parameters: {ceta_T}, {T}, {TN}, {period}, {K}')
        deltaHN = ZeroBond.H(TN) - ZeroBond.H(T)
        print(f'deltaHN: {deltaHN}')
        tau = TAUS[period]
        time_add = TIMES[period]
        num_times = int(np.float64((TN - T) / time_add))
        x0 = 0
        y_star = fsolve(fystar, x0)
        print(f'y_star: {y_star}')
        print(f'Function result: {fystar(y_star)}')
        # First term
        first_term = ZeroBond.D(T) * norm.cdf(- y_star/np.sqrt(ceta_T), 0, 1)
        # Second term
        second_term = - ZeroBond.D(TN) * norm.cdf(- (y_star + deltaHN*ceta_T)/np.sqrt(ceta_T), 0, 1)
        # Third term
        third_term = 0
        for i in range(1, num_times + 1):
            Ti = T + i * time_add
            deltaHi = ZeroBond.H(Ti) - ZeroBond.H(T)
            third_term += ZeroBond.D(Ti) * norm.cdf( - (y_star + deltaHi * ceta_T) / np.sqrt(ceta_T), 0, 1)
        print(f'Final i: {i}, {Ti}')
        third_term *= - tau * K
        return first_term + second_term + third_term

    
    @staticmethod
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
        fixed *= - tau * K
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
        fixed *= - tau * K
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
    
class TestExamples():
    
    @staticmethod
    def test_function(x: np.array, t: np.array, T: int, r: float, sigma: float):
        difference = T - t
        exponential = np.exp((r + sigma)*difference)
        norma = np.linalg(x, axis = 0)
        return exponential * norma
        
    @staticmethod
    def strike_value(x: tf.Tensor):
        return tf.linalg.norm(x)