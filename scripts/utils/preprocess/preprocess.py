import pandas as pd
import numpy as np

class Preprocessor:
    @staticmethod
    def preprocess_paths(
        T,
        N_steps,
        mc_paths_transpose,
        nsims
    ):
        dts = list(np.linspace(0, T, N_steps)) * len(mc_paths_transpose)
        simulation = [
            [i] * N_steps for i in range(nsims)
        ]
        df_x_tmp = pd.DataFrame(
            zip(
                mc_paths_transpose,
                simulation
            ),
            columns = [
                'path',
                'simulation'
            ]
        )
        df_x = pd.DataFrame()
        df_x['xt'] = df_x_tmp.explode('path')['path']
        df_x['dt'] = dts
        df_x['simulation'] = df_x_tmp.explode('simulation')['simulation']
        df_x['_delta_x'] = df_x.groupby([
            'simulation',
        ])['xt'].shift(1)
        df_x['_delta_x'] = (df_x['xt'] - df_x['_delta_x'])
        df_x.loc[df_x._delta_x.isna(), '_delta_x'] = 0.
        # Sort to get the examples in blocks
        df_x.sort_values(
            [
                'simulation',
                'dt'
            ],
            inplace = True
        )
        
        return df_x