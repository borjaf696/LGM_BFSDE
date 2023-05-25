import os
import numpy as np
import pandas as pd
from abc import (
    ABC,
    abstractmethod
)
from utils.utils.utils import (
    ZeroBond,
    IRS,
    Swaption,
    FinanceUtils
)

class Tester(ABC):
    
    def _calculate_basics(self, df, model, sigma_value = None):
    
        mc_paths_tranformed = df[['xt', 'dt']].values
        x = mc_paths_tranformed.astype(np.float64)
        delta_x = df._delta_x.values.astype(np.float64)
        v_lgm_single_step, _ = model.predict(
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
        # Calculate C and N
        t_unique = df.dt.unique()
        dict_C = {dt:FinanceUtils.C(dt, sigma_value = sigma_value) for dt in t_unique}
        df['ct'] = df.apply(lambda x: dict_C[x['dt']], axis = 1)
        df['N'] = ZeroBond.N(
            df.dt.values.astype(np.float64),
            df.xt.values.astype(np.float64),
            df.ct.values.astype(np.float64)
        )
        df['V_est'] = df.lgm_single_step_V * df.N

        return df
    
    @abstractmethod
    def test(self, df, model, test_name_file, sigma_value, TM = None, T = None):
        pass
    
class ZeroBondTester(Tester):
    
    def test(self, df, model, test_name_file, sigma_value, TM = None, T = None):
        assert test_name_file is not None, 'Test name file is not provided'
        
        df = super()._calculate_basics(
            df, 
            model,  
            sigma_value
        )
        df['V'] = ZeroBond.Z(
            df.xt.values.astype(np.float64),
            df.dt.values.astype(np.float64),
            np.float64(df.dt.max()),
            df.ct.values.astype(np.float64)
        )
        
        folder = '/'.join(
            test_name_file.split('/')[:-1]
        )
        print(f'{folder}')
        if not os.path.exists(folder):
            print(f'Creating folder {folder}')
            os.makedirs(folder)
        
        df.to_csv(
            test_name_file, 
            index = False
        )
        
        
class IRSTester(Tester):
    
    def test(self, df, model, test_name_file, sigma_value, TM = None, T = None):
        assert test_name_file is not None, 'Test name file is not provided'
        
        df = super.__calculate_basics(
            df, 
            model, 
            test_name_file, 
            sigma_value
        )
        print(f'Please provide the IRS analytical formula')
        
        folder = '/'.join(
            test_name_file.split('/')[:-1]
        )
        print(f'{folder}')
        if not os.path.exists(folder):
            print(f'Creating folder {folder}')
            os.makedirs(folder)
            
        df.to_csv(
            test_name_file, 
            index = False
        )
        
class SwaptionTester(Tester):
    
    def test(self, df, model, test_name_file, sigma_value, TM = None, T = None):
        assert test_name_file is not None, 'Test name file is not provided'
        
        df = super.__calculate_basics(
            df, 
            model, 
            test_name_file, 
            sigma_value
        )
        
        df['V'] = Swaption.Swaption_test(
            df.xt.values.astype(np.float64),
            df.dt.values.astype(np.float64),
            T,
            TM,
            df.ct.values.astype(np.float64)
        )
        
        folder = '/'.join(
            test_name_file.split('/')[:-1]
        )
        print(f'{folder}')
        if not os.path.exists(folder):
            print(f'Creating folder {folder}')
            os.makedirs(folder)
            
        df.to_csv(
            test_name_file, 
            index = False
        )