import pandas as pd
import numpy as np


class Preprocessor:
    @staticmethod
    def preprocess_paths(T, N_steps, mc_paths_transpose, nsims):
        dts = list(np.linspace(0, T, N_steps)) * len(mc_paths_transpose)
        simulation = [[i] * N_steps for i in range(nsims)]
        df_x_tmp = pd.DataFrame(
            zip(mc_paths_transpose, simulation), columns=["path", "simulation"]
        )
        df_x = pd.DataFrame()
        df_x["xt"] = df_x_tmp.explode("path")["path"]
        df_x["dt"] = dts
        df_x["simulation"] = df_x_tmp.explode("simulation")["simulation"]
        df_x["_delta_x"] = df_x.groupby(
            [
                "simulation",
            ]
        )[
            "xt"
        ].shift(1)
        df_x["_delta_x"] = df_x["xt"] - df_x["_delta_x"]
        df_x.loc[df_x._delta_x.isna(), "_delta_x"] = 0.0
        # Sort to get the examples in blocks
        df_x.sort_values(["simulation", "dt"], inplace=True)
        # Remove unused dfs
        del df_tmp

        return df_x

    @staticmethod
    def preprocess_paths_multidimensional(T, N_steps, dim, mc_paths, nsims):
        dts = list(np.linspace(0, T, N_steps)) * mc_paths.shape[1]
        simulation = []
        for sim in range(nsims):
            simulation += [sim] * N_steps
        df_x = pd.DataFrame(zip(dts, simulation), columns=["dt", "sim"])
        for i in range(dim):
            df_x[f"X_{i}"] = mc_paths[:, :, i].T.reshape((N_steps * nsims, 1))
            df_x[f"_delta_x_{i}"] = df_x.groupby(["sim"])[f"X_{i}"].shift(1)
            df_x[f"delta_x_{i}"] = df_x[f"X_{i}"] - df_x[f"_delta_x_{i}"]
            df_x.loc[df_x[f"delta_x_{i}"].isna(), f"delta_x_{i}"] = 0.0
        # Sort to get the examples in blocks
        df_x.sort_values(["sim", "dt"], inplace=True)

        return df_x
