# Linear Gaussian Markov Model for Forward Backward Stochastic Differential Equations

This project focuses on employing both shallow and deep neural networks to model and adjust complex financial products. The core concept revolves around defining an objective function, $\phi$ which is articulated through the equation $\hat{v}_{t+1} = v_t + \frac{\partial N}{\partial x_t}\cdot \delta_x$. In this equation, $\frac{\partial N}{\partial x_t}$represents the derivative of the network with respect to $x_t$, and $v_t$  represents the network's prediction at time $t$. It is easy to see that $\frac{\partial N}{\partial x_t}$ represents and approximation of $\frac{\partial \phi}{\partial x_t}$.

A significant challenge in the financial domain is that the analytical formulae for many financial products at a given time $t$ are either unknown or too complex to compute efficiently. Typically, only the expression of the function at maturity is known. To address this, the loss function utilized in this project is:

$$
\mathcal{L}_{path} = \frac{1}{n-1}\sum_{i = 1}^{n - 1}\left(\hat{v}_i - v_i\right)^2 + (\phi(x_T) - v_T)^2 + (\frac{\partial\phi}{\partial x_T} - \frac{\partial NN}{\partial x_T})^2
$$

This loss function clearly indicates that the function is not known at any given time $t$, but only at maturity $T$. Consequently, the problem is framed as an semisupervised learning task.

In essence, this approach leverages neural networks to infer the dynamics of financial products from limited information. By focusing on the function at maturity and using a tailored loss function, it aims to predict and adjust the behavior of complex financial instruments, a task that traditional analytical methods may find challenging due to the lack of explicit formulae or the computational complexity involved. The combination of shallow and deep neural networks potentially offers a flexible and powerful toolset for modeling these intricate financial structures.

## Installation

Follow these steps to get **LGM-FBSDE** up and running on your system.

### Prerequisites

* Operating System: Windows 10 or later, macOS X 10.14 or later, or a recent distribution of Linux.
* Python version 3.7 or higher
* More than 4 Gb of RAM (GPU cuda capable is suggested).

### Step-by-step installation guide

##### Download the repository from GitHub

```bash
git clone git@github.com:borjaf696/LGM-BFSDE.git
```

##### Set Up a Python Virtual environment, to do so I suggest using `pyenv`.

The installation steps are from **mac** but for linux or Windows is similar.

* Install `pyenv .`

```bash
brew update
brew install pyenv
```

* Configure the shell.

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

* Install a python distribution.

```bash
pyenv install 3.9.0 lgm-fbsde
```

* Set the local distribution in your project folder.

```bash
pyenv local lgm-fbsde
```

* Install `hatch` as environment manager:

```bash
pip install hatch
```

* Install the environment with `hatch`

```bash
hatch env create
```

### Weights and Biases (Wandb) Integration

To run the experiments we have used "wandb". To install it simply run:

```
hatch run pip install wandb
```

In the code there is an entry `wand.login()` which will ask you to login in into wandb but just the first time you use it.

## Execution

To test the correct installation we can run a simple example by doing:

```bash
hatch run python scripts/launcher.py --normalize True --T 1 --phi zerobond --nsteps 48 --nsims 100 --nepochs 10 --sigma 0.01 --schema 1 --wandb True --save True
```

In this toy_test we are running 100 simulations for 10 epochs, thus the results are not going to be impressive but it will allow you to test the correctness of the installation.

For more examples please check the **experiments** folder.

## Current Capabilities of the Code: Handling Diverse Financial Products

Our current implementation of the neural network-based system is equipped to handle four distinct types of financial products, each with its unique characteristics and complexities. These products are:

1. **Zero-Coupon Bond (ZCB):**
   * A zero-coupon bond is a debt security that doesn't pay interest (a coupon) but is traded at a deep discount, rendering profit at maturity when the bond is redeemed for its full face value.
   * The simplicity of its structure makes it a suitable candidate for modeling with neural networks, as it provides a clear understanding of how discount factors and interest rates interact over time.
2. **Interest Rate Swap (IRS):**
   * An IRS is a forward contract in which one stream of future interest payments is exchanged for another based on a specified principal amount.
   * IRS contracts are fundamental instruments in financial markets, used for hedging and speculating on the movement of interest rates.
3. **Swaption:**
   * A swaption is an option granting its owner the right but not the obligation to enter into an underlying IRS.
   * This derivative is more complex due to its optionality feature, requiring sophisticated models to approximate its valuation and risk characteristics.
4. **Custom-Defined Function at Maturity (TODO):**
   * The framework is being designed to accommodate custom-defined functions at maturity. This feature will allow users to input their unique payoff functions for financial products at the time of maturity.
   * **TODO:** This functionality is currently under development. Our goal is to create a flexible system where users can define and integrate their bespoke financial instruments or payoff structures into the neural network model.

Each of these products poses unique challenges and requires careful consideration in their modeling. By employing both shallow and deep neural networks, we aim to capture the nuanced behaviors of these instruments and accurately predict their future values under various market conditions. The versatility in handling different product types makes this system particularly valuable for financial analysts and institutions looking to leverage machine learning for financial forecasting and analysis.
