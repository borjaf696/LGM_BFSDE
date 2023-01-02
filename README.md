# Linear Gaussian Markov Model for Forward Backward Stochastic Differential Equations

The repository contains the code developed to prove the viability of neural networks to correctly predict the value of a *swap option* at any given *t*

## Theory

## How to:

Logic:
* F - neural network function.
* $\frac{\delta F}{\delta X_t}^i$ - gradient calculated by using the model at $i$-iteration.
* $\phi(n, x_n)$ - known terminal function.

$$\hat{V} = F(X)$$

$$\overline{V}_0 = \overline{V}^i[0]$$

$$\overline{V}_{t+1} = \overline{V}_t + \frac{\delta F(X)}{\delta x_t}(x_{t + 1} - x_{t})$$

$$\mathcal{L}(\overline{V}, \hat{V}) = \beta_1 \cdot (\hat{V}_n - \phi(n, x_n))^2 + \beta_2\cdot (\hat{V}_n - \frac{\delta F(X)}{\delta x_n})^2 + \sum_{i = 1}^{n - 1}(\overline{V}_i - \hat{V}_i)^2  $$

## Todo:

* Implement using Tensorflow the feed forward architecture to given a sequence $X = \{x_1, x_2, \dots, x_n\}$ predict a sequence $Y = \{V_1, V_2, \dots, V_n\}$. 
* Add the custom loss function.
* Implement the logic.
* How are we going to evaluate this?