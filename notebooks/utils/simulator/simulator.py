# Regular imports
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
from operator import itemgetter
from scipy import stats
from typing import Optional


class MCSimulation:
    def __init__(self, T, N, X0, sigma, model="LGM"):
        self._params = {
            "T": T,
            "N": N,
            "X0": X0,
            "sigma": sigma,
        }
        self._model = model

    def simulate(self, nsim=1e3, show=False):
        T, N, X0, sigma = itemgetter("T", "N", "X0", "sigma")(self._params)
        nsim = int(nsim)
        dt = T / N
        # Brownian simulation
        W, X = np.zeros([N, nsim]), np.zeros([N, nsim])
        # Starting point
        W[0, :] = X0
        X[0, :] = X0
        for i in range(1, N):
            W[i, :] = W[i - 1, :] + np.random.randn(nsim) * np.sqrt(dt)
        # X simulation
        for i in range(1, N):
            X[i, :] = X[i - 1, :] + sigma * (W[i, :] - W[i - 1, :])
        if show:
            X = np.linspace(0, T, N)

        return X, W
