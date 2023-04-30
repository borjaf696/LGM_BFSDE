from typing import Any
from model.model_lgm_single_step import LGM_model_one_step

import numpy as np
import pickle as pkl
import pandas as pd
import time 

def trainer(
        epochs: int = 50,
        size_of_the_batch: int = 100,
        *,
        T: int,
        N_steps: int,
        sigma: float,
        nsims: int,
        phi: Any,
        phi_str: str,
        df_x: pd.DataFrame,
):
    model_name = 'models_store/' + phi_str + '_model_lgm_single_step.h5'
    # Fixed for now
    epochs = 50
    # Batch execution with baby steps
    size_of_the_batch = 100
    batch_size = size_of_the_batch * N_steps
    batches = int(np.floor(nsims * N_steps / batch_size))
    try:
        with open(model_name, 'rb') as f:
           lgm_single_step = pkl.load(f)
        return lgm_single_step
    except FileNotFoundError: 
        print(f'{model_name} not found')
    print(f'Training {epochs} epochs with {size_of_the_batch} paths per epoch with length {N_steps}')
    # Custom model
    lgm_single_step = LGM_model_one_step(n_steps = N_steps, 
                                     T = T, 
                                     future_T = 2 * T,
                                     verbose = False,
                                     sigma = sigma,
                                     batch_size = size_of_the_batch,
                                     phi = phi,
                                     name = 'irs'
    )
    print(f'{lgm_single_step.summary()}')
    # Compile the model
    lgm_single_step.define_compiler(
        optimizer = 'adam', 
        learning_rate = 1e-3
    )
    # Data used as features
    mc_paths_tranformed = df_x[['xt', 'dt']].values
    x = mc_paths_tranformed.astype(np.float64)
    delta_x = df_x._delta_x.values.astype(np.float64)
    # Custom iteration: 
    for epoch in range(epochs):
        print(f'{epoch}...', end = '')
        for batch in range(batches):
            start_time = time.time()
            x_batch = x[batch * batch_size: (batch + 1) * batch_size, :]
            delta_x_batch = delta_x[batch * batch_size: (batch + 1) * batch_size]
            lgm_single_step.custom_train_step(
                X = x_batch,
                batch = batch,
                epoch = epoch, 
                start_time = start_time,
                delta_x = delta_x_batch
            )
    # Pickling the model
    with open(model_name, 'wb') as f:
        pkl.dump(lgm_single_step, f)
        print(f'Model saved in {model_name}')
        
    return lgm_single_step