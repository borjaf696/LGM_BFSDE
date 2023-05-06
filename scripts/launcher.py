import os
import argparse
# TODO: Remove asap
import numpy as np
import pandas as pd

from utils.preprocess.preprocess import Preprocessor
from utils.simulator.simulator import MCSimulation
from utils.utils.utils import (
    ZeroBond,
    IRS,
    Swaption
)

from trainer.trainer import trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Descripción del programa")
    
    parser.add_argument("-p", "--phi", type=str, help="Phi function to be used irs/swaption/zerobond (default zerobond)", default = 'zerobond')
    parser.add_argument("-n", "--nsims", type=str, help="Number of simulations (default 1000)", default=100)
    parser.add_argument("-t", "--T", type=int, help="Strike time 1/2/4/8 (default 1)", default=1)
    parser.add_argument("-s", "--sigma", type=float, help="Active volatility (default 10%)", default=0.01)
    parser.add_argument("-q", "--nsteps", type=float, help="Number of steps for each path (default 100)", default=100)
    parser.add_argument("--test", type=bool, help="Test", default = True)
    args = parser.parse_args()
    
    return args

def get_phi(active):
    phi = None
    if active == 'zerobond':
        phi = ZeroBond.Z_normalized
    elif active == 'irs':
        phi = IRS.IRS_normalized
    elif active == 'swaption':
        phi = Swaption.Swaption_normalized
    return phi

# TODO: Move to a tester
def test(df, model, test_name_file = None):
    assert test_name_file is not None, 'Test name file is not provided'
    
    mc_paths_tranformed = df[['xt', 'dt']].values
    x = mc_paths_tranformed.astype(np.float64)
    delta_x = df._delta_x.values.astype(np.float64)
    v_lgm_single_step, predictions = model.predict(
        x, 
        delta_x,
        build_masks = True
    )
    # TODO: denormalize
    results = pd.DataFrame(
        zip(v_lgm_single_step),
        columns = ['results']
    )
    v_lgm_single_step = results.explode('results').values
    df['lgm_single_step_V'] = v_lgm_single_step.astype(np.float64)
    # Export to file
    df.to_csv(
        test_name_file, 
        index = False,
        sep = ';'
    )
    print(f'Results saved to: {test_name_file}')
    
if __name__ == '__main__':
    args = parse_args()
    # Configs
    T, N_steps, X0, sigma = (
        args.T, 
        args.nsteps,
        0, 
        args.sigma
    )
    nsims = args.nsims
    print(f'Simulations:')
    print(f'\tNsims: {nsims}')
    print(f'\tT: {T}')
    print(f'\tN_steps: {N_steps}')
    print(f'\tX0: {X0}')
    print(f'\tSigma: {sigma}')
    mcsimulator = MCSimulation(
        T, 
        N_steps, 
        X0, 
        sigma
    )
    # Training paths
    mc_paths, W = mcsimulator.simulate(nsims)
    mc_paths_transpose = mc_paths.T
    df_x = Preprocessor.preprocess_paths(
        T,
        N_steps,
        mc_paths_transpose,
        nsims
    )
    print(f'Features shape: {df_x.shape}')
    # tf model training
    phi_str = args.phi
    # Imprime los valores de los argumentos
    print(f"Function selected: {args.phi}")
    # Get the environment
    phi = get_phi(phi_str)
    print(f"Training: {phi}")
    # Train the model
    model = trainer(
        T = T,
        N_steps = N_steps,
        sigma = sigma,
        nsims = nsims,
        phi = phi,
        phi_str = phi_str,
        df_x = df_x
    )
    # Test
    if args.test:
        # TODO: Remove from here
        test_name_file = 'data/export/' + phi_str + '_test_results_' + str(T) + '_' + str(nsims)+ '.csv'
        test_sims = int(nsims * 0.2)
        mc_paths_test, W_test = mcsimulator.simulate(test_sims)
        mc_paths_transpose_test = mc_paths_test.T
        df_x_test = Preprocessor.preprocess_paths(
            T,
            N_steps,
            mc_paths_transpose_test,
            test_sims
        )
        print(f'Test Features shape: {df_x_test.shape}')
        # Data used as features
        test(
            df = df_x_test,
            model = model,
            test_name_file = test_name_file
        )