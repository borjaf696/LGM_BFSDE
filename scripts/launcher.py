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
    Swaption,
    FinanceUtils
)

from utils.tester.tester import (
    SwaptionTester,
    ZeroBondTester,
    IRSTester
)

from trainer.trainer import trainer

# Exception (TODO: Move to a custom file)
class ArgumentFailure(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje
    def __str__(self):
        return self.mensaje

# TODO: Reformat
def parse_args():
    parser = argparse.ArgumentParser(description="Descripción del programa")
    
    parser.add_argument("--phi", type=str, help="Phi function to be used irs/swaption/zerobond (default zerobond)", default = 'zerobond')
    parser.add_argument("--TM", type = int, help = "Time to end Swap/IRS (default 8)", default = None)
    parser.add_argument("--T", type=int, help="Strike time 1/2/4/8 (default 1)", default=1)
    parser.add_argument("--nsims", type=int, help="Number of simulations (default 1000)", default=1000)
    parser.add_argument("--sigma", type=float, help="Active volatility (default 10%)", default=0.01)
    parser.add_argument("--nsteps", type=float, help="Number of steps for each path (default 100)", default=100)
    parser.add_argument("--test", type=bool, help="Test", default = True)
    args = parser.parse_args()
    
    if args.TM is None and args.phi != 'zerobond':
        raise ArgumentFailure('If {args.phi} then --TM must be specified')
        
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

def get_phi_test(active):
    phi = None
    if active == 'zerobond':
        phi = ZeroBondTester()
    elif active == 'irs':
        phi = IRSTester()
    elif active =='swaption':
        phi = SwaptionTester()
    return phi
    
if __name__ == '__main__':
    args = parse_args()
    # Configs
    T, TM, N_steps, X0, sigma = (
        args.T, 
        args.TM,
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
        T = T, 
        N = N_steps, 
        X0 = X0, 
        sigma = sigma
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
    print(f"Training: {phi_str}")
    # Train the model
    model = trainer(
        T = T,
        TM = TM,
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
        mcsimulator = MCSimulation(
            T = T,  
            X0 = X0, 
            sigma = sigma,
            N = N_steps,
            period = None
        )
        mc_paths_test, W_test = mcsimulator.simulate(
            test_sims
        )
        mc_paths_transpose_test = mc_paths_test.T
        df_x_test = Preprocessor.preprocess_paths(
            T,
            mcsimulator.N,
            mc_paths_transpose_test,
            test_sims
        )
        print(f'Test Features shape: {df_x_test.shape}')
        print(f'Test columns: {df_x_test.columns}')
        tester = get_phi_test(phi_str)
             
        # Data used as features
        tester.test(
            df = df_x_test,
            model = model,
            test_name_file = test_name_file,
            # TODO: Deprecate this
            sigma_value = sigma,
            TM = args.TM,
            T = T
        )