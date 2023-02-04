# Linear Gaussian Markov Model for Forward Backward Stochastic Differential Equations

The repository contains the code developed to prove the viability of neural networks to correctly predict the value of a *swap option* at any given *t*

## Theory

Logic:
* F - neural network function.
* $\frac{\delta F}{\delta X_t}^i$ - gradient calculated by using the model at $i$-iteration.
* $\phi(n, x_n)$ - known terminal function.

$$\hat{V} = F(X)$$

$$\overline{V}_0 = \overline{V}^i[0]$$

$$\overline{V}_{t+1} = \overline{V}_t + \frac{\delta F(X)}{\delta x_t}(x_{t + 1} - x_{t})$$

$$\mathcal{L}(\overline{V}, \hat{V}) = \beta_1 \cdot (\hat{V}_n - \phi(n, x_n))^2 + \beta_2\cdot (\hat{V}_n - \frac{\delta F(X)}{\delta x_n})^2 + \sum_{i = 1}^{n - 1}(\overline{V}_i - \hat{V}_i)^2  $$

## Evaluation

The evaluation of each step is better expained in ***notebooks/simulation.ipynb***

* Evaluation of LGM Montecarlo simulations


The final objective is to check that $E[-\frac{1}{2}H_T^2\zeta_t-H_Tx_t] = 0$

* Second evaluation - evaluate loss function and NN work.

In this particular case we have ground truth since we have the analytical expression for Zero Bound Coupon. Therefore, we can use this expresion to compare against what NN suggests.

* Third evaluation (and model selection) - define the optimal model architecture.

#### How to run

Install the environment. The next step will build a folder ***.venv*** which stores a python environment, the requirements and dependencies are stored directly in ***pyproject.toml***:
<pre><code> pip install hatch</code></pre>
<pre><code> hatch run python</code></pre>

You are good to run the notebooks stored in ***notebooks/***

## Todo:

- Check Loss function.
- Check the gradient tape.
- Sanity with Zero bond coupon.