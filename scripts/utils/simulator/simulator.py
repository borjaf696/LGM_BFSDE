# Regular imports
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
from operator import itemgetter
from scipy import stats
from typing import Optional

from scripts.utils.utils.utils import TIMES


class MCSimulation:
    def __init__(self, T, X0, sigma, N=None, dim=None, period=None, model="LGM"):
        self._params = {
            "T": T,
            "N": N if period is None else int(T / TIMES[period]),
            "dim": dim,
            "X0": X0,
            "sigma": sigma,
        }
        self._model = model

    @property
    def N(self):
        return self._params["N"]

    def simulate(self, nsim=1e3, show=False):
        T, N, X0, sigma, dim = itemgetter("T", "N", "X0", "sigma", "dim")(self._params)
        nsim = int(nsim)
        dt = T / N
        # Brownian simulation
        dimensions = [N, nsim, dim]
        W, X = np.zeros(dimensions), np.zeros(dimensions)
        # Starting point
        W[0, :, :] = X0
        X[0, :, :] = X0
        for i in range(1, N):
            W[i, :, :] = W[i - 1, :, :] + np.random.randn(nsim, dim) * np.sqrt(dt)
        # X simulation
        for i in range(1, N):
            X[i, :, :] = X[i - 1, :, :] + sigma * (W[i, :] - W[i - 1, :])
        if show:
            X = np.linspace(0, T, N)
        print(f"[MEMORY] Simulation memory usage: {(W.nbytes + X.nbytes) / 2**30} Gb")
        return X, W
