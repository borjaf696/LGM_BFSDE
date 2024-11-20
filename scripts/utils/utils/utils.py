import sys
import warnings

import tensorflow as tf
import numpy as np
import pandas as pd

from scipy.stats import norm
from typing import Any, List, Dict

# Constants
TAUS = {3: 0.25, 6: 0.5, 12: 1.0}
TIMES = {3: 0.25, 6: 0.5, 12: 1.0}

# TODO: Change the IRS/Zerobond/Swaption to products


# Cyclic learning rate with decay
class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, max_lr, step_size, decay=1.0, mode="triangular"):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.decay = decay
        self.mode = mode

    def __call__(self, step):
        cycle = tf.floor(1. + tf.cast(step, tf.float32) / (2 * self.step_size))
        x = tf.abs(tf.cast(step, tf.float32) / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * tf.maximum(0., (1 - x))
        return lr * (self.decay ** (tf.floor(tf.cast(step, tf.float32) / self.step_size)))

class TFUtils:
    @staticmethod
    def custom_reshape(tensor):
        tensor_shape = tf.shape(tensor)
        tensor_rank = tf.size(tensor_shape)
        reshaped_tensor = tf.cond(
            tf.equal(tensor_rank, 1),
            lambda: tf.reshape(tensor, [-1, 1]),
            lambda: tensor,
        )
        return reshaped_tensor

    @staticmethod
    def safe_divide(numerator, denominator, value_if_zero=0.0):
        denominator_safe = tf.where(
            denominator == 0, tf.ones_like(denominator), denominator
        )
        result = numerator / denominator_safe
        result_safe = tf.where(
            denominator == 0, tf.ones_like(result) * value_if_zero, result
        )
        return result_safe


class Utils:

    @staticmethod
    def get_features_with_pattern(
        df: pd.DataFrame, pattern: str, extra_cols: List[str]
    ):
        features = []
        for column in df.columns.values:
            if column[0] == pattern:
                features.append(column)
        features += extra_cols

        return features

    @staticmethod
    def print_progress_bar(iteration, total, prefix="", suffix="", length=50, fill="█"):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)
        sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
        sys.stdout.flush()


class MathUtils:

    @staticmethod
    def custom_derivative_model(x, model, multioutput=True):
        h = 1e-3
        # Dimensions
        x_dim, y_dim, z_dim = x.shape
        # Gradient vector
        gradient = np.zeros((x_dim, y_dim, z_dim))
        for i in range(z_dim):
            # Vector for partial derivative estimation
            offset_tensor = np.zeros((x_dim, y_dim, 1))
            offset_tensor[:, i] = h
            offset_tensor = tf.convert_to_tensor(offset_tensor, dtype=tf.float64)
            # Constantes:
            denominator = h
            numerator = tf.math.subtract(
                model(tf.math.add(x, offset_tensor)),
                model(tf.math.subtract(x, offset_tensor)),
            )
            denominator = 2 * h
            gradient[:, :, i] = tf.reshape(numerator / denominator, (x_dim, y_dim))
        gradient = tf.convert_to_tensor(gradient, dtype=tf.float64)
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

        gradients = tf.convert_to_tensor(gradients, dtype=tf.float32)

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
                offset_tensor = tf.convert_to_tensor(offset_tensor, dtype=tf.float64)
                # Constantes:
                denominator = h
                numerator = tf.math.subtract(
                    model(tf.math.add(x, offset_tensor)),
                    model(tf.math.subtract(x, offset_tensor)),
                )
                denominator = 2 * h
                gradient[:, j, i] = numerator[:, j, 0] / denominator
        gradient = tf.convert_to_tensor(gradient, dtype=tf.float64)
        return gradient


class MLUtils:
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin

    ## TODO:
    # ------ Adapt fit to work only with training and then transform given the
    # ------ the previous transformation
    class NormalizeColumns(BaseEstimator, TransformerMixin):
        """S-normalization

        Args:
            BaseEstimator (_type_): _description_
            TransformerMixin (_type_): _description_
        """

        def __init__(self, columns=None):
            self.columns = columns
            self.__sufix = "_transformed"

        def fit(self, X, y=None):
            self.__mean = np.mean(X, axis=0)
            self.__std = np.std(X, axis=0)
            return self

        def transform(self, X, y=None):
            cols_to_transform = list(X.columns)

            if self.columns:
                cols_to_transform = self.columns
            cols_renamed = []
            for column in cols_to_transform:
                cols_renamed.append(column + self.__sufix)

            X[cols_renamed] = (X[cols_to_transform] - self.__mean) / (self.__std)
            return X

    def sin_activation(x):
        return tf.math.sin(x)

    @staticmethod
    def get_pipeline(transformations=None):
        from sklearn.pipeline import Pipeline

        transformation_map = {"normalization": MLUtils.NormalizeColumns()}
        pipeline_steps = []
        for transformation in transformations:
            pipeline_steps.append((transformation, transformation_map[transformation]))
        pipe = Pipeline(steps=pipeline_steps)
        return pipe


class FinanceUtils:
    @staticmethod
    def sigma(t, sigma_0=0.0075):
        """_summary_

        Args:
            t (_type_): _description_
            sigma_0 (float, optional): _description_. Defaults to 0.0075.

        Returns:
            _type_: _description_
        """
        return sigma_0 + (t // 2) * 0.0005

    @staticmethod
    def C(t, sigma_value=0.0075):
        """_summary_

        Args:
            t (_type_): _description_
            sigma (_type_, optional): _description_. Defaults to sigma.

        Returns:
            _type_: _description_
        """
        from scipy.integrate import quad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return quad(
                lambda x: FinanceUtils.sigma(t=x, sigma_0=sigma_value) ** 2,
                a=0,
                b=t,
                limit=50,
            )[0]


class ZeroBond:
    @staticmethod
    def H_tensor(t, kappa=0.02):
        """_summary_

        Args:
            t (_type_): _description_
            kappa (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        return (1 - tf.math.exp(-kappa * t)) / kappa

    @staticmethod
    def D_tensor(t, r=0.03):
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
        return tf.math.multiply(
            1 / ZeroBond.D_tensor(t),
            tf.math.exp(
                tf.math.add(
                    tf.math.multiply(ZeroBond.H_tensor(t), xt),
                    0.5 * tf.math.square(ZeroBond.H_tensor(t)) * ct,
                )
            ),
        )

    @staticmethod
    def Z_tensor(xt, t, T, ct=None):
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
        discount_factor = ZeroBond.D_tensor(T)
        first_exponential_term = -ZeroBond.H_tensor(T) * xt
        second_exponential_term = -0.5 * ZeroBond.H_tensor(T) ** 2 * ct
        N = ZeroBond.N_tensor(t, xt, ct)
        return tf.math.multiply(
            discount_factor,
            tf.math.multiply(
                tf.math.exp(
                    tf.math.add(first_exponential_term, second_exponential_term)
                ),
                N,
            ),
        )

    @staticmethod
    def D(t, r=0.03):
        """_summary_

        Args:
            t (_type_): _description_
            r (float, optional): _description_. Defaults to 0.03.

        Returns:
            _type_: _description_
        """
        return np.exp(-r * t)

    @staticmethod
    def H(t, kappa=0.02):
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
        return (
            1
            / ZeroBond.D(t)
            * np.exp(ZeroBond.H(t) * xt + 0.5 * ZeroBond.H(t) ** 2 * ct)
        )

    @staticmethod
    def Z(xt, t, T, ct=None):
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
        return (
            ZeroBond.D(T)
            * np.exp(-0.5 * ZeroBond.H(T) ** 2 * ct - ZeroBond.H(T) * xt)
            * ZeroBond.N(t, xt, ct)
        )

    @staticmethod
    def Z_norm(xn, tn, T, ct=None):
        """_summary_
        Args:
            xn (_type_): _description_
            n (_type_): _description_
            ct (_type_): _description_
        Returns:
            _type_: _description_
        """
        assert ct is not None
        return ZeroBond.Z(xn, tn, T, ct) / ZeroBond.N(tn, xn, ct)

    @staticmethod
    def exponent(xt, t, T, ct=None):
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
        res = tf.math.multiply(
            ZeroBond.Z_tensor(xn, tn, T, ct), 1 / ZeroBond.N_tensor(tn, xn, ct)
        )
        return tf.reshape(res, tf.shape(xn))

    @staticmethod
    def Z_strike_normalized(xn, T, Tm, ct, period=None, K=None):
        """_summary_
        Args:
            xn (_type_): _description_
            n (_type_): _description_
            ct (_type_): _description_
        Returns:
            _type_: _description_
        """
        return tf.math.multiply(
            ZeroBond.Z_tensor(xn, T, T, ct), 1 / ZeroBond.N_tensor(T, xn, ct)
        )


class Swap:
    @staticmethod
    def anuality_swap(xn, t, T, TN, ct, period=6):
        tau = tf.cast(TAUS[period], dtype=tf.float64)
        time_add = tf.cast(TIMES[period], dtype=tf.float64)
        num_times = tf.cast((TN - T) / time_add + 1, dtype=tf.int64)
        indices = tf.range(1, num_times, dtype=tf.float64)

        def body(i):
            return ZeroBond.Z_normalized(xn, t, T + i * time_add, ct)

        fixed_contributions = tf.map_fn(body, indices, fn_output_signature=tf.float64)
        fixed = tf.reduce_sum(fixed_contributions, axis=0)
        return tau * fixed

    @staticmethod
    def par_swap(
        xn,
        t,
        Ti,
        Tm,
        ct,
        period=6,
    ):
        pi = ZeroBond.Z_normalized(xn, t, Ti, ct)
        pm = ZeroBond.Z_normalized(xn, Ti, Tm, ct)
        # For anuality we only require last Zeta_t
        anuality = Swap.anuality_swap(xn, Ti, Ti, Tm, ct, period)
        return tf.math.multiply((pi - pm), 1 / anuality)

    @staticmethod
    def positive_parswap(xn, t, Ti, Tm, ct, period=6, K=0.03):
        par_swap = Swap.par_swap(xn, t, Ti, Tm, ct, period)
        return np.maximum(0, par_swap - K)

    @staticmethod
    def positive_parswap_tf(xn, Ti, Tm, ct, period=6, K=0.03, nominal=10000, smoothed=True, alpha = tf.constant(0.001, dtype = tf.float64)):
        par_swap = Swap.par_swap(xn, Ti, Ti, Tm, ct, period)
        nominal_tf = tf.constant(nominal, dtype=tf.float64)
        K_tf = tf.constant(K, dtype=tf.float64)
        if tf.rank(par_swap) == 0:
            try:
                par_swap = tf.reshape(par_swap, (1,tf.shape(par_swap)[0]))
            except Exception as e:
                par_swap = tf.reshape(par_swap, (1,1))
        if smoothed:
            soft_max = tf.math.log(
                tf.reduce_sum(
                    tf.concat(
                        [
                            tf.exp(
                                (tf.reshape(par_swap, (tf.shape(par_swap)[0], 1)) - K_tf) / alpha
                            ),
                            tf.exp(tf.zeros((tf.shape(par_swap)[0], 1), dtype=par_swap.dtype)),
                        ],
                        axis=1,
                    ),
                    axis=1,
                )
            )
            return nominal_tf * soft_max
        else:
            return nominal_tf * tf.reduce_max(
                tf.concat(
                    [
                        tf.reshape(par_swap, (tf.shape(par_swap)[0], 1)) - K_tf,
                        tf.zeros((tf.shape(par_swap)[0], 1), dtype=par_swap.dtype),
                    ],
                    axis=1,
                ),
                axis=1,
            )

    @staticmethod
    def density_normal(x, mu, var):
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (x - mu) ** 2 / var)


class IRS:
    @staticmethod
    def IRS(xn, T, TN, ct, period=6, K=0.03):
        # A(i,m)
        anuality_term = Swap.anuality_swap(xn, T, TN, ct, period)
        # Par Swap
        par_swap = Swap.par_swap(xn, T, T, TN, ct, period)
        # Numeraire
        N = ZeroBond.N_tensor(T, xn, ct)
        # Compose the IRS result
        return N * anuality_term * (par_swap - K)

    @staticmethod
    def IRS_normalized(xn, T, Tm, ct, period=6, K=0.03, nominal = 1):
        # A(i,m)
        anuality_term = Swap.anuality_swap(xn, T, T, Tm, ct, period)
        # Par Swap
        par_swap = Swap.par_swap(xn, T, T, Tm, ct, period)
        # Numeraire
        N = ZeroBond.N_tensor(T, xn, ct)
        # Compose the IRS result
        return nominal * N * anuality_term * (par_swap - K) / N

    @staticmethod
    def IRS_test_normalized(xn, t, Ti, Tm, ct, period=6, K=0.03):
        # A(i,m)
        anuality_term = Swap.anuality_swap(xn, t, Ti, Tm, ct, period)
        # Par Swap
        par_swap = Swap.par_swap(xn, t, Ti, Tm, ct, period)
        # Numeraire
        N = ZeroBond.N_tensor(t, xn, ct)
        # Compose the IRS result
        return N * anuality_term * (par_swap - K) / N

    # TODO: Adapt
    @staticmethod
    def IRS_test(xn, t, Ti, Tm, ct, period=6, K=0.03, nominal = 1):
        # A(i,m)
        anuality_term = Swap.anuality_swap(xn, t, Ti, Tm, ct, period)
        # Par Swap
        par_swap = Swap.par_swap(xn, t, Ti, Tm, ct, period)
        # Numeraire
        N = ZeroBond.N_tensor(t, xn, ct)
        # Compose the IRS result
        return nominal * N * anuality_term * (par_swap - K)


class Swaption:

    @staticmethod
    def Swaption_test(xn, t, Ti, Tm, ct, period=6, K=0.03, cT=None):
        def integra_swap(xT, xnj, ct, cT):
            positive_par_swap = Swap.positive_parswap_tf(
                xn=xT, Ti=Ti, Tm=Tm, ct=cT, period=period, K=K
            )
            # print(f"Positive parswaps: {positive_par_swap}")
            density_normal = Swap.density_normal(x=xT, mu=xnj, var=cT - ct)
            return positive_par_swap * density_normal

        def swaption_at_strike(xn, ct):
            positive_par_swap = Swap.positive_parswap_tf(
                xn=xn, Ti=Ti, Tm=Tm, ct=ct, period=period, K=K
            )
            return float(positive_par_swap)

        import scipy.integrate as integrate

        cT = tf.reduce_max(ct) if cT is None else cT
        swaption_results = []
        for i, t_local in enumerate(t):
            xni = xn[i]
            cti = ct[i]
            if t_local != Ti:
                swaption_results.append(
                    integrate.fixed_quad(
                        integra_swap,
                        xni - 4 * np.sqrt(cT - cti),
                        xni + 4 * np.sqrt(cT - cti),
                        n=100,
                        args=(xni, cti, cT),
                    )[0]
                )
            else:
                swaption_results.append(swaption_at_strike(xni, cti))
        N = ZeroBond.N_tensor(t, xn, ct)
        # anuality_term = Swap.anuality_swap(xn, t, Ti, Tm, ct, period)
        return swaption_results * N # * anuality_term

    def Swaption_test_tf(xn, t, Ti, Tm, ct, period=6, K=0.03, cT=None):
        cT = tf.reduce_max(ct) if cT is None else cT
        swaption_results = []

        for i, t_local in enumerate(t):
            xni = xn[i]
            cti = ct[i]

            if t_local != Ti:
                x_min = xni - 4 * tf.sqrt(cT - cti)
                x_max = xni + 4 * tf.sqrt(cT - cti)
                x_values = tf.linspace(x_min, x_max, 100)

                def integra_swap(xT):
                    positive_par_swap = Swap.positive_parswap_tf(
                        xn=xT, Ti=Ti, Tm=Tm, ct=cT, period=period, K=K
                    )
                    density_normal = Swap.density_normal(x=xT, mu=xni, var=cT - cti)
                    anuality_term = Swap.anuality_swap(xT, Ti, Tm, cT, period)
                    return positive_par_swap * anuality_term * density_normal

                def swaption_at_strike(xn, ct):
                    positive_par_swap = Swap.positive_parswap_tf(
                        xn=xn, Ti=Ti, Tm=Tm, ct=ct, period=period, K=K
                    )
                    anuality_term = Swap.anuality_swap(xn, Ti, Tm, ct, period)
                    return positive_par_swap * anuality_term

                integral_result = tf.map_fn(
                    lambda x: integra_swap(x),
                    x_values,
                    dtype=(tf.float64),
                    parallel_iterations=100,
                )
                integral_result = tf.reduce_sum(integral_result)
                swaption_results.append(integral_result)
            else:
                swaption_results.append(
                    swaption_at_strike(xni, cti, Ti, Tm, period, K, cT)
                )

        N = ZeroBond.N_tensor(t, xn, ct)
        return swaption_results * N

    @staticmethod
    def Swaption_test_normalized(xn, t, Ti, Tm, ct, period=6, K=0.03):
        N = ZeroBond.N_tensor(t, xn, ct)
        return Swaption.Swaption_test(xn, t, Ti, Tm, ct, period, K) / N

    @staticmethod
    def Swaption_test_normalized_tf(xn, t, Ti, Tm, ct, period=6, K=0.03):
        N = ZeroBond.N_tensor(t, xn, ct)
        return Swaption.Swaption_test_tf(xn, t, Ti, Tm, ct, period, K) / N

    @staticmethod
    def Swaption_at_zero(
        ceta_T,
        T,
        TN,
        period=6,
        K=0.03,
    ):
        from scipy.optimize import fsolve
        import numpy as np

        def fystar(x):
            # First term
            exponential_first_term = -deltaHN * x
            exponential_second_term = -0.5 * deltaHN**2 * ceta_T
            first_term = ZeroBond.D(TN) * np.exp(
                exponential_first_term + exponential_second_term
            )
            # Second term
            second_term = 0
            for i in range(1, num_times + 1):
                Ti = T + i * time_add
                deltaHi = ZeroBond.H(Ti) - ZeroBond.H(T)
                exponential_first_term_second = -deltaHi * x
                exponential_second_term_second = -0.5 * deltaHi**2 * ceta_T
                second_term += ZeroBond.D(Ti) * np.exp(
                    exponential_first_term_second + exponential_second_term_second
                )
            second_term *= tau * K
            # Third term
            third_term = -ZeroBond.D(T)
            return first_term + second_term + third_term
        deltaHN = ZeroBond.H(TN) - ZeroBond.H(T)
        tau = TAUS[period]
        time_add = TIMES[period]
        num_times = int(np.float64((TN - T) / time_add))
        x0 = 0
        y_star = fsolve(fystar, x0)
        # First term
        first_term = ZeroBond.D(T) * norm.cdf(-y_star / np.sqrt(ceta_T), loc=0, scale=1)
        # Second term
        second_term = -ZeroBond.D(TN) * norm.cdf(
            -(y_star + deltaHN * ceta_T) / np.sqrt(ceta_T), loc=0, scale=1
        )
        # Third term
        third_term = 0
        for i in range(1, num_times + 1):
            Ti = T + i * time_add
            deltaHi = ZeroBond.H(Ti) - ZeroBond.H(T)
            third_term += ZeroBond.D(Ti) * norm.cdf(
                -(y_star + deltaHi * ceta_T) / np.sqrt(ceta_T), loc=0, scale=1
            )
        third_term *= -tau * K
        return first_term + second_term + third_term

    @staticmethod
    def Swaption_at_zero_int(
        ceta_T,
        T,
        TN,
        period=6,
        K=0.03,
    ):
        def integra_zero(y):
            time_add = TIMES[period]
            num_times = int(np.float64((TN - T) / time_add))
            anuality_term = 0
            for i in range(1, num_times + 1):
                Ti = T + i * time_add
                deltaHi = ZeroBond.H(Ti)
                deltaHi_sq = ZeroBond.H(Ti) ** 2
                anuality_term += (
                    TAUS[period]
                    * K
                    * ZeroBond.D(Ti)
                    * np.exp(-deltaHi * y - 0.5 * deltaHi_sq * ceta_T)
                )
            deltaHN = ZeroBond.H(TN)
            deltaHN_sq = ZeroBond.H(TN) ** 2
            final_pay_term = ZeroBond.D(TN) * np.exp(
                -deltaHN * y - 0.5 * deltaHN_sq * ceta_T
            )
            first_pay_term = ZeroBond.D(T) * np.exp(
                -ZeroBond.H(T) * y - 0.5 * ZeroBond.H(T) ** 2 * ceta_T
            )
            normal_density = Swap.density_normal(y, mu=0, var=ceta_T)
            res = -anuality_term - final_pay_term + first_pay_term
            res[res < 0] = 0
            return res * normal_density

        import scipy.integrate as integrate

        return integrate.fixed_quad(
            integra_zero,
            0 - 4 * np.sqrt(ceta_T),
            0 + 4 * np.sqrt(ceta_T),
            n=100,
        )[0]

    @staticmethod
    def Swaption(xn, T, Tm, ct, period=6, K=0.03):
        return Swaption.Swaption_test(xn, T, T, Tm, ct, period, K)

    @staticmethod
    def Swaption_normalized(xn, T, Tm, ct, period=6, K=0.03):
        positive_par_swap = Swap.positive_parswap_tf(
            xn=xn, Ti=T, Tm=Tm, ct=ct, period=period, K=K
        )
        # anuality_term = Swap.anuality_swap(xn, T, T, Tm, ct, period)
        return positive_par_swap # * anuality_term


class TestExamples:

    @staticmethod
    def toy_test_function(x: np.array, t: np.array, T: int, r: float, sigma: float):
        difference = T - t
        exponential = np.exp((r + sigma) * difference)
        norma = np.linalg(x, axis=0)
        return exponential * norma

    @staticmethod
    def strike_test(xn, T, Tm, ct, period=None, K=None):
        return tf.math.exp(xn)



class GPUUtils:
    
    @staticmethod
    def set_device(device = "cpu", gpu_number = 1):
        if device.upper() == "CPU":
            tf.config.set_visible_devices([], "GPU")
            print("[DEVICE] Using CPU")
        elif device.upper() == "GPU":
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    # Use the first GPU found
                    tf.config.set_visible_devices(gpus[gpu_number], "GPU")
                    logical_gpus = tf.config.list_logical_devices("GPU")
                    print(f"[DEVICE] Physical GPU, {len(logical_gpus)}, Logical GPU")
                    print(f"[DEVICE] Set device GPU: {gpus[gpu_number]}")
                except RuntimeError as e:
                    print(e)
            else:
                print("[DEVICE] No GPU found, using CPU instead.")
                tf.config.set_visible_devices([], "GPU")
        else:
            raise ValueError("[DEVICE] Unrecognized device. Use 'CPU' or 'GPU'.")