# Linear Gaussian Markov Model for Forward Backward Stochastic Differential Equations

The repository contains the code developed to prove the viability of neural networks to correctly predict the value of a *swap option* at any given *t*

## Theory

## How to:

## Todo:

    * Implement using Tensorflow the feed forward architecture to given a sequence $X = \{x_1, x_2, \dots, x_n\}$ predict a sequence $Y = \{V_1, V_2, \dots, V_n\}$. 
    * Implement the logic:
        * F - neural network function
        * $\frac{\delta F}{\delta X_t}^i$ - gradient calculated by using the model at $i$-iteration
        $$V^i = F^i(X)$$
        $$\overline{V}_0^i = \overline{V}^i[0]$$
        $$\overline{V}^{i + 1}_{t+1} = \overline{V}_t + \frac{\delta \overline{V}_t^i}{\delta x_t}$$