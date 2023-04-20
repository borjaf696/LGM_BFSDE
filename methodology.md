## TODO:

Define the whole theoretical background:

* Brownian motion
* Ito
* LGM model
* Finance models

#### MonteCarlo Simulation

The MonteCarlo simulation is going to be used to generate random paths, Brownian paths. 

Mathematical speaking a Brownian path is a stochastic continuous in time and with increments independent and equally distributed, $\mathcal{N}(0, \Delta t)$. It can be represented as:

$$
dX_t = \mu dt + \sigma dW_t
$$

where:

* $X_t$ is a position of a particle, value of a stock, etc, at a particular $t$ point
* $\mu$ is the expected change (bias).
* $\sigma$ is the stock volatility or the desviation in the expected change.
* $W_t$ is a Wiener process. 

Furthermore:

$$
dW_t = \sqrt{\Delta t}\cdot \epsilon_t
$$

with $\epsilon_t \sim \mathcal{N}(0, 1)$

Therefore, the distribution of $W_t$ can be written as:

$$
W_t \sim\mathcal{N}(0,t)
$$

The simulation process then is done as:

$$
W_0 = S_0
$$

$$
W_t = W_{t - 1}  + dWt = W_{t-1} + \sqrt{\Delta t}\cdot \epsilon_t
$$

Then we simulate our path from the Wiener process:

$$
X_{t + 1} = X_t + \sigma\cdot (W_{t + 1} - W_t) = X_t + \sigma\cdot\Delta W
$$

$$
\Delta W = W_{t+1} - W_t
$$

We have our paths $X$ which we are going to use to train our models to be able to learn the value of the option at any given $t$.

### Linear Gaussian Markov model

The Gaussian Markov Model (LGM) is a statistical model used in finance to model the evolution of interest rates and other financial factors over time. It is based on the theory of stochastic processes and is used to estimate the volatility and correlation of financial factors, allowing investors and risk managers to make informed decisions about portfolio management.

Why Markov? 

* The model assumes a probability of changes of the financial factors only depends on the current value and not in all the previous values. Therefore,

  $$
  P(X_{t+1}| X_t, \dots, X_{t - n}) = P(X_{t+1}|X_t)
  $$

  In this case, $X$ is the distribution of changes of the value through time.

Why Linear?

* The model assumes that the changes in the different values are proportional to the current values. So basically it assumes that the relation between the current values of the factors and the changes is lineal.

$$
y_t = \beta_0 + \beta_1 x_{1,t} + \beta_2 x_{2,t} + \cdots + \beta_k x_{k,t} + \epsilon_t
$$

LGM in our context:

$$
dx_t = \sigma_tsW_t^N

$$

In financial context is pretty common to define a "market" normalization value typically known as **numeraire:**


$$
N(t, x_t) = \frac{1}{B(0,t)}exp^{H_tx_t + \frac{1}{2}H_t^2\zeta_t}
$$

For two known functions: $H_t$ and $\zeta_t$.

Let's define then the value of our option as $V_t=V(x_t, t)$ and the deflated version as:

$$
\overline{V}_t = \frac{V_t}{N_t}
$$


## Theory

* F - neural network function.
* $\frac{\delta F}{\delta X_t}^i$ - gradient calculated by using the model at $i$-iteration.
* $\phi(n, x_n)$ - known terminal function.

$$
\hat{V} = F(X)
$$

$$
\overline{V}_0 = \overline{V}^i[0]
$$

$$
\overline{V}_{t+1} = \overline{V}_t + \frac{\delta F(X)}{\delta x_t}(x_{t + 1} - x_{t})
$$

$$
\mathcal{L}(\overline{V}, \hat{V}) = \beta_1 \cdot (\hat{V}_n - \phi(n, x_n))^2 + \beta_2\cdot (\hat{V}_n - \frac{\delta F(X)}{\delta x_n})^2 + \sum_{i = 1}^{n - 1}(\overline{V}_i - \hat{V}_i)^2
$$

## Implementation

## Evaluation

The evaluation of each step is better expained in ***notebooks/simulation.ipynb***

* Evaluation of LGM Montecarlo simulations

The final objective is to check that $E[-\frac{1}{2}H_T^2\zeta_t-H_Tx_t] = 0$

* Second evaluation - evaluate loss function and NN work.

In this particular case we have ground truth since we have the analytical expression for Zero Bound Coupon. Therefore, we can use this expresion to compare against what NN suggests.

* Third evaluation (and model selection) - define the optimal model architecture.
