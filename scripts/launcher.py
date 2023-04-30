import os
import argparse

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
    parser.add_argument("-n", "--nsims", type=str, help="Number of simulations (default 1000)", default=1000)
    parser.add_argument("-t", "--T", type=int, help="Strike time 1/2/4/8 (default 1)", default=1)
    parser.add_argument("-s", "--sigma", type=float, help="Active volatility (default 10%)", default=0.01)
    parser.add_argument("-q", "--nsteps", type=float, help="Number of steps for each path (default 100)", default=100)
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
    print(f'Features shape: {x.shape}')
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
    