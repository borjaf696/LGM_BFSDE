# Linear Gaussian Markov Model for Forward Backward Stochastic Differential Equations

The repository contains the code developed to prove the viability of neural networks to correctly predict the value of a *swap option* at any given *t*

## How to run

Install the environment. The next step will build a folder ***.venv*** which stores a python environment, the requirements and dependencies are stored directly in ***pyproject.toml***:

<pre><code> pip install hatch</code></pre>

<pre><code> hatch run python</code></pre>

You are good to run the notebooks stored in ***notebooks/***

## Todo:

- [ ] MC simulation
  - [ ] Brownian motion paths
- [ ] Define Neural Network models:
  - [ ] Single step model
  - [ ] Sequence model
  - [ ] ...
- [ ] Sanities:
  - [ ] ZeroBond
  - [ ] IRS
  - [ ] ...
