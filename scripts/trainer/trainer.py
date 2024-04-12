from typing import Any

# LGM models
# TODO: Change this (factory)
from scripts.model.lgm_adjusted import LgmSingleStepModelAdjusted
from scripts.model.lgm_naive import LgmSingleStepNaive
from scripts.model.lgm_naive_time_adjust import LgmSingleStepNaiveTimeAdjust

# Simulator
from scripts.utils.simulator.simulator import MCSimulation
from scripts.utils.preprocess.preprocess import Preprocessor
from scripts.utils.utils.utils import CyclicLR, Utils

# Basics python
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import time
import tracemalloc
import os, psutil

# Wandb integration
import wandb
from wandb.keras import WandbCallback


def simulate(T, N_steps, dim, sigma, nsims):
    mcsimulator = MCSimulation(T=T, N=N_steps, dim=dim, X0=0, sigma=sigma)
    # Training paths
    mc_paths, w = mcsimulator.simulate(nsims)
    if len(mc_paths.shape) < 2:
        df_x = Preprocessor.preprocess_paths(T, N_steps, mc_paths, nsims)
    else:
        df_x = Preprocessor.preprocess_paths_multidimensional(
            T, N_steps, dim, mc_paths, nsims
        )
    features = []
    for column in df_x.columns.values:
        if column[0] == "X":
            features.append(column)
    features.append("dt")
    # Remove unused data
    del mc_paths, w

    return df_x, features


def trainer(
    epochs: int = 110,
    size_of_the_batch: int = 100,
    architecture: str = "ff_dense_single_step",
    *,
    T: int,
    TM: int,
    N_steps: int,
    dim: int,
    sigma: float,
    nsims: int,
    phi: Any,
    phi_str: str,
    normalize: bool,
    report_to_wandb: bool = False,
    anneal_lr: bool = True,
    schema: int = 1,
    save_model: bool = False,
    simulate_in_epoch: bool = False,
    device: str = "cpu",
):
    # Start tracer
    tracemalloc.start()
    if report_to_wandb:
        wandb.init(
            project="lgm",
            name=f"schema_{schema}_normalize_{normalize}_{phi_str}_{T}_{TM}_{epochs}_{size_of_the_batch}_{nsims}_{N_steps}_{architecture}_dim_{dim}",
            config={
                "epochs": epochs,
                "size_of_the_batch": size_of_the_batch,
                "T": T,
                "TM": TM,
                "N_steps": N_steps,
                "sigma": sigma,
                "nsims": nsims,
                "phi": phi,
                "phi_str": phi_str,
                "architecture": architecture,
                "schema": schema,
                "normalize": normalize,
                "dim": dim,
                "log_step": 10,
            },
        )
    # Fixed for now
    epochs = epochs
    # Batch execution with baby steps
    size_of_the_batch = 64
    # Recalculate nsims
    batch_size = size_of_the_batch * N_steps
    batches = int(np.floor(nsims * N_steps / batch_size))
    # LGM model instance
    future_T = TM if TM is not None else T
    # Initial simulation to adapt normalization
    df_x, features = simulate(
        T=T,
        N_steps=N_steps,
        dim=dim,
        sigma=sigma,
        nsims=min(10 * nsims, 100000) if simulate_in_epoch else nsims,
    )
    x0 = df_x[features].values.astype(np.float64)

    if schema == 1:
        lgm_single_step = LgmSingleStepNaive(
            n_steps=N_steps,
            T=T,
            future_T=future_T,
            dim=x0.shape[1],
            verbose=False,
            sigma=sigma,
            batch_size=size_of_the_batch,
            phi=phi,
            name=phi_str,
            report_to_wandb=report_to_wandb,
            normalize=normalize,
            data_sample=x0,
            device=device,
        )
    elif schema == 2:
        lgm_single_step = LgmSingleStepModelAdjusted(
            n_steps=N_steps,
            T=T,
            future_T=future_T,
            dim=dim,
            verbose=False,
            sigma=sigma,
            batch_size=size_of_the_batch,
            phi=phi,
            name=phi_str,
            report_to_wandb=report_to_wandb,
            normalize=normalize,
            data_sample=x0,
        )
    elif schema == 3:
        lgm_single_step = LgmSingleStepNaiveTimeAdjust(
            n_steps=N_steps,
            T=T,
            future_T=future_T,
            dim=dim,
            verbose=False,
            sigma=sigma,
            batch_size=size_of_the_batch,
            phi=phi,
            name=phi_str,
            report_to_wandb=report_to_wandb,
            normalize=normalize,
            data_sample=x0,
        )
    print(
        f"[MEMORY] Main dataframe memory usage: {df_x.memory_usage(deep = True).sum() / 2**30} Gb"
    )
    # Model name
    if TM is not None:
        model_name = f"models_store/{phi_str}_{schema}_model_lgm_single_sigma_{sigma}_dim_{dim}_normalize_{normalize}_step_{T}_{TM}_{nsims}_{N_steps}_epochs_{epochs}_batchsize_{size_of_the_batch}.weights.h5"
    else:
        model_name = f"models_store/{phi_str}_{schema}_model_lgm_single_sigma_{sigma}_dim_{dim}_normalize{normalize}_step_{T}_{nsims}_{N_steps}_epochs_{epochs}_batchsize_{size_of_the_batch}.weights.h5"
    # lgm_single_step.export_model_architecture()
    try:
        lgm_single_step.load_weights(model_name)
        return lgm_single_step
    except FileNotFoundError:
        print(f"{model_name} not found")
    print(
        f"Training {epochs} epochs with {size_of_the_batch} paths per epoch with length {N_steps}"
    )
    print(f"{lgm_single_step.summary()}")
    # Starting learning rate
    if anneal_lr:
        lr = CyclicLR(
            base_lr=1e-3, max_lr=0.006, step_size=100, decay=0.99, mode="triangular"
        )
    else:
        lr = 3e-5
    # Compile the model
    lgm_single_step.define_compiler(optimizer="adam", learning_rate=lr)
    # Custom iteration:
    epoch = 0
    loss = np.infty

    while loss > 0.00001 and epoch < epochs:
        # Data used as features
        if (epoch == 0) or simulate_in_epoch:
            if simulate_in_epoch:
                df_x, features = simulate(
                    T=T, N_steps=N_steps, dim=dim, sigma=sigma, nsims=nsims
                )
            mc_paths_tranformed = df_x[features].values
            x = mc_paths_tranformed.astype(np.float64)
            delta_x = df_x.delta_x_0.values.astype(np.float64)
            # Create the tf.Dataset
            x = tf.reshape(
                x, (nsims, N_steps, 2)
            )  # Num simulations, steps per simulation, t and x
            delta_x = tf.reshape(delta_x, (nsims, N_steps))
            dataset = tf.data.Dataset.from_tensor_slices((x, delta_x))
            dataset = dataset.batch(size_of_the_batch)
        print(f"{epoch}...", end="")
        snapshot1 = tracemalloc.take_snapshot()
        for batch, (x_batch, delta_x_batch) in enumerate(dataset):
            Utils.print_progress_bar(
                batch, batches, prefix="batches", suffix="|", length=50, fill="█"
            )
            local_batch_size = x_batch.shape[0]
            x_batch = tf.reshape(x_batch, (N_steps * local_batch_size, 2))
            delta_x_batch = tf.reshape(delta_x_batch, (N_steps * local_batch_size, 1))

            process = psutil.Process(os.getpid())
            memory_use = process.memory_info().rss / (1024 * 1024)
            # print(f"\n\tMemory usage_1 (trainer): {memory_use}")
            x = tf.Variable(x_batch, trainable=True, dtype=tf.float64)
            delta_x = tf.Variable(delta_x_batch, trainable=False, dtype=tf.float64)
            loss, _, _, _ = lgm_single_step.custom_train_step_tf(
                x=x,
                delta_x=delta_x,
            )
            process = psutil.Process(os.getpid())
            memory_use = process.memory_info().rss / (1024 * 1024)
            # print(f"\tMemory usage_2 (trainer): {memory_use}")
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        print("[ Top 10 diferencias ]")
        for stat in top_stats[:10]:
            print(stat)
        if epoch % 1 == 0:
            lgm_single_step.plot_tracker_results(epoch)
        epoch += 1
        # Reset error trackers
        lgm_single_step.reset_trackers()
        # Clear tf backend
        lgm_single_step.clear_memory()
    # Explanation cut training
    print(f"Finishing training with {epoch} epochs and loss {loss:.4f}")
    # Save the model
    if save_model:
        lgm_single_step.save_weights(model_name)

    return lgm_single_step
