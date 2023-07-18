from typing import Any
from scripts.model.model_lgm_single_step import LGM_model_one_step

import numpy as np
import pickle as pkl
import pandas as pd
import time 

# Wandb integration
import wandb
from wandb.keras import WandbCallback

from utils.utils.utils import CyclicLR

def trainer(
        epochs: int = 110,
        size_of_the_batch: int = 100,
        architecture: str = 'ff_dense_single_step',
        *,
        T: int,
        TM: int,
        N_steps: int,
        sigma: float,
        nsims: int,
        phi: Any,
        phi_str: str,
        df_x: pd.DataFrame,
        report_to_wandb: bool = False,
        anneal_lr: bool = True
):
    if wandb:
        wandb.init(
            project="lgm",
            name = f'{phi_str}_{T}_{TM}_{epochs}_{size_of_the_batch}_{nsims}_{N_steps}_{architecture}',
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
                "log_step": 10
            }
        )
    if TM is not None:
        model_name = 'models_store/' + phi_str + '_model_lgm_single_step' + str(T) + '_' + str(TM) + '_' + str(nsims) + '_' + str(N_steps)+ '.h5'
    else:
        model_name = 'models_store/' + phi_str + '_model_lgm_single_step' + str(T) + '_' + str(nsims)+ '_' + str(N_steps) + '.h5'
    # Fixed for now
    epochs = epochs
    # Batch execution with baby steps
    size_of_the_batch = 100
    batch_size = size_of_the_batch * N_steps
    batches = int(np.floor(nsims * N_steps / batch_size))
    # Data used as features
    mc_paths_tranformed = df_x[['xt', 'dt']].values
    x = mc_paths_tranformed.astype(np.float64)
    delta_x = df_x._delta_x.values.astype(np.float64)
    # LGM model instance
    future_T = TM if TM is not None else T
    lgm_single_step = LGM_model_one_step(n_steps = N_steps, 
                                     T = T, 
                                     future_T = future_T,
                                     verbose = False,
                                     sigma = sigma,
                                     batch_size = size_of_the_batch,
                                     phi = phi,
                                     name = phi_str,
                                     report_to_wandb=report_to_wandb,
                                     data = x
    )
    lgm_single_step.export_model_architecture()
    try:
        lgm_single_step.load_weights(model_name)
        return lgm_single_step
    except FileNotFoundError: 
        print(f'{model_name} not found')
    print(f'Training {epochs} epochs with {size_of_the_batch} paths per epoch with length {N_steps}')
    print(f'{lgm_single_step.summary()}')
    # Starting learning rate
    if anneal_lr:
        lr = CyclicLR(base_lr=1e-3, max_lr=0.006, step_size=10, decay=0.99, mode='triangular')
    else:
        lr = 1e-3
    # Compile the model
    lgm_single_step.define_compiler(
        optimizer = 'adam', 
        learning_rate = lr
    )
    # Custom iteration: 
    epoch = 0
    loss = np.infty
    while loss > 0.00001 and epoch < epochs:
        print(f'{epoch}...', end = '')
        for batch in range(batches):
            start_time = time.time()
            x_batch = x[batch * batch_size: (batch + 1) * batch_size, :]
            delta_x_batch = delta_x[batch * batch_size: (batch + 1) * batch_size]
            loss, _, _, _ = lgm_single_step.custom_train_step(
                X = x_batch,
                batch = batch,
                epoch = epoch, 
                start_time = start_time,
                delta_x = delta_x_batch
            )
        epoch += 1
        # Reset error trackers
        lgm_single_step.reset_trackers()
    # Explanation cut training
    print(f'Finishing training with {epoch} epochs and loss {loss:.4f}')
    # Save the model
    lgm_single_step.save_weights(model_name)
        
    return lgm_single_step