# Linear Gaussian Markov Model for Forward Backward Stochastic Differential Equations

The repository contains the code developed to prove the viability of neural networks to correctly predict the value of a *swap option* at any given *t*

## How to run

Install the environment. The next step will build a folder ***.venv*** which stores a python environment, the requirements and dependencies are stored directly in ***pyproject.toml***:

<pre><code> pip install hatch</code></pre>

<pre><code> hatch run python</code></pre>

#### Launcher

To launch the code:

- `parámetro1`:
- `parámetro2`:
- `parámetro3`:

### Weights and Biases (Wandb) Integration

To run the experiments we have used "wandb". To install it simply run:

```
hatch run pip install wandb
```

In the code there is an entry `wand.login()` which will ask you to login in into wandb but just the first time you use it.

## Todo:

- [X] Sanities:
  - [X] ZeroBond
  - [X] IRS
  - [X] Swaption
