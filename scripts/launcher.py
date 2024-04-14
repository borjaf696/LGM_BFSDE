import sys
import os

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)
print(f"Project root: {project_root}")
import argparse

# TODO: Remove asap
import numpy as np
import pandas as pd

# Simulator
from utils.simulator.simulator import MCSimulation
from utils.utils.utils import GPUUtils
from utils.preprocess.preprocess import Preprocessor

from utils.utils.utils import ZeroBond, IRS, Swaption, TestExamples, Utils

from utils.tester.tester import SwaptionTester, ZeroBondTester, IRSTester

from trainer.trainer import trainer

import wandb
from wandb.keras import WandbCallback


# Exception (TODO: Move to a custom file)
class ArgumentFailure(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje

    def __str__(self):
        return self.mensaje


# TODO: Reformat
def parse_args():
    parser = argparse.ArgumentParser(description="Descripción del programa")

    parser.add_argument(
        "--phi",
        type=str,
        help="Phi function to be used irs/swaption/zerobond (default zerobond)",
        default="zerobond",
    )
    parser.add_argument(
        "--TM", type=int, help="Time to end Swap/IRS (default 8)", default=None
    )
    parser.add_argument(
        "--T", type=int, help="Strike time 1/2/4/8 (default 1)", default=1
    )
    parser.add_argument(
        "--nsims", type=int, help="Number of simulations (default 500)", default=500
    )
    parser.add_argument(
        "--dim", type=int, help="Number of dimensions for each component", default=1
    )
    parser.add_argument(
        "--sigma", type=float, help="Active volatility (default 10%)", default=0.01
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        help="Number of steps for each year path (default 100)",
        default=50,
    )
    parser.add_argument(
        "--simulate-in-epoch",
        type=bool,
        help="Simulate paths in each epoch",
        default=False,
    )
    # Schema
    parser.add_argument("--schema", type=int, help="Schema of the model", default=1)
    # Trainer
    parser.add_argument(
        "--normalize", type=bool, help="Do normalization", default=False
    )
    parser.add_argument(
        "--nepochs", type=int, help="Number of epochs (default 100)", default=100
    )
    parser.add_argument("--save", type=bool, help="Save the model", default=False)
    parser.add_argument("--device", type=str, help="Device to be used", default="cpu")
    # Wandb Tracker
    parser.add_argument("--wandb", type=bool, help="Wandb", default=False)
    # Test
    parser.add_argument(
        "--test", type=bool, help="Do Testing", default=True
    )
    args = parser.parse_args()

    if args.TM is None and args.phi != "zerobond":
        raise ArgumentFailure("If {args.phi} then --TM must be specified")

    return args


def get_phi(active):
    phi = None
    if active == "zerobond":
        phi = ZeroBond.Z_strike_normalized
    elif active == "irs":
        phi = IRS.IRS_normalized
    elif active == "swaption":
        phi = Swaption.Swaption_normalized
    elif active == "test":
        phi = TestExamples.strike_test
    return phi


def get_phi_test(active):
    phi = None
    if active == "zerobond":
        phi = ZeroBondTester()
    elif active == "irs":
        phi = IRSTester()
    elif active == "swaption":
        phi = SwaptionTester()
    return phi


if __name__ == "__main__":
    args = parse_args()
    # Wandb integration
    if args.wandb:
        wandb.login()

    # Phi function
    phi = get_phi(args.phi)
    if phi is None:
        raise ArgumentFailure(f"{args.phi} is not a valid phi function")
    # Configs
    T, TM, N_steps, X0, sigma, dim = (
        args.T,
        args.TM,
        args.nsteps * args.T,
        0,
        args.sigma,
        args.dim,
    )
    nsims = args.nsims
    print(f"Simulations:")
    print(f"\tNsims: {nsims}")
    print(f"\tT: {T}")
    print(f"\tTM: {TM}")
    print(f"\tN_steps: {N_steps}")
    print(f"\tDimensions: {dim}")
    print(f"\tX0: {X0}")
    print(f"\tSigma: {sigma}")
    print(f"\tDevice: {args.device}")
    print(f"\tLaunch test: {args.test}")
    # tf model training
    phi_str = args.phi
    # Imprime los valores de los argumentos
    print(f"Function selected: {args.phi}")
    # Get the environment
    phi = get_phi(phi_str)
    print(f"Training: {phi_str}")
    # Normalize:
    normalize = args.normalize
    print(f"Normalize: {normalize}")
    # Simulate per epoch
    simulate_per_epoch = args.simulate_in_epoch
    print(f"Simulate per epoch: {simulate_per_epoch}")
    # Schema selected:
    print(f"Schema: {args.schema}")
    # Set the device
    GPUUtils.set_device(args.device)
    # Epochs
    epochs = args.nepochs
    # Train the model
    model = trainer(
        epochs=epochs,
        T=T,
        TM=TM,
        N_steps=N_steps,
        dim=dim,
        sigma=sigma,
        nsims=nsims,
        phi=phi,
        phi_str=phi_str,
        normalize=normalize,
        report_to_wandb=args.wandb,
        schema=args.schema,
        save_model=args.save,
        simulate_in_epoch=args.simulate_in_epoch,
        device = args.device
    )
    # Test
    if args.test:
        # TODO: Remove from here
        test_name_file = f"data/export/test/{phi_str}_{args.schema}_normalize_{normalize}_test_results_sigma_{sigma}_dim_{dim}_{T}_{TM}_{nsims}_{N_steps}_epochs_{epochs}.csv"
        # TODO: Remove from here
        train_name_file = f"data/export/train/{phi_str}_{args.schema}_normalize_{normalize}_train_results_sigma_{sigma}_dim_{dim}_{T}_{TM}_{nsims}_{N_steps}_epochs_{epochs}.csv"
        test_sims = min(int(nsims * 0.001), 10000)
        mcsimulator = MCSimulation(
            T=T, X0=X0, sigma=sigma, N=N_steps, dim=dim, period=None
        )
        mc_paths_test, W_test = mcsimulator.simulate(test_sims)
        if len(mc_paths_test.shape) < 2:
            df_x_test = Preprocessor.preprocess_paths(
                T, N_steps, mc_paths_test, test_sims
            )
        else:
            df_x_test = Preprocessor.preprocess_paths_multidimensional(
                T, N_steps, dim, mc_paths_test, test_sims
            )
        features = Utils.get_features_with_pattern(
            df=df_x_test, pattern="X", extra_cols=["dt"]
        )
        print(f"Test Features shape: {df_x_test.shape}")
        print(f"Test columns: {df_x_test.columns}")
        tester = get_phi_test(phi_str)

        # Data used as features
        tester.test(
            df=df_x_test,
            model=model,
            test_name_file=test_name_file,
            features=features,
            # TODO: Deprecate this
            sigma_value=sigma,
            TM=TM,
            T=T,
            report_to_wandb=args.wandb,
        )
