import tensorflow as tf
import numpy as np

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
    def C(t, sigma = sigma):
        """_summary_

        Args:
            t (_type_): _description_
            sigma (_type_, optional): _description_. Defaults to sigma.

        Returns:
            _type_: _description_
        """
        import scipy.integrate as integrate
        return integrate.quad(lambda x: sigma(x)**2, 0, t)[0]    
    
    @staticmethod
    @tf.function
    def zero_bond_coupon(xn, tn, ct):
        """_summary_

        Args:
            xn (_type_): _description_
            n (_type_): _description_
            ct (_type_): _description_

        Returns:
            _type_: _description_
        """
        return tf.math.multiply((1.0 + xn*0.), 1/ ZeroBound.N_tensor(tn, xn, ct))
    
    def zero_bond_coupon_np(xn, tn, ct):
        """_summary_

        Args:
            xn (_type_): _description_
            n (_type_): _description_
            ct (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (1.0 + xn * 0.) / ZeroBound.N(tn, xn, ct)
    
class ZeroBound():  
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
        return tf.math.multiply(1/ZeroBound.D_tensor(t), tf.math.exp(tf.math.add(tf.math.multiply(ZeroBound.H_tensor(t), xt), 0.5 * tf.math.square(ZeroBound.H_tensor(t)) * ct)))
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
        return 1/ZeroBound.D(t) * np.exp(ZeroBound.H(t) * xt + 0.5 * ZeroBound.H(t)**2 * ct)
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
        return ZeroBound.D(T) * np.exp(-0.5 * ZeroBound.H(T)**2 * ct - ZeroBound.H(T)*xt) * ZeroBound.N(t, xt, ct)