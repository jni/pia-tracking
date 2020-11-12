from annotate_tracks import CostEvaluation
from btrack_tracking import track_objects
import dask.array as da 
from data_io import single_zarr
import functools
import json
import optuna 
import pandas as pd
from _parser import get_paths, custom_parser, track_view_base
from random_tracks_engine import RandomTracksEngine
import time
from timeout import timeout
from toolz import curry


# -----------------------------------------------------------------------------
# Build Config
# ------------

def build_config(
                 dt=1.0,
                 measurements=3,
                 states=6,
                 accuracy=7.5,
                 prob_not_assign=0.01,
                 max_lost=5,
                 A_matrix=[1,0,0,1,0,0,
                           0,1,0,0,1,0,
                           0,0,1,0,0,1,
                           0,0,0,1,0,0,
                           0,0,0,0,1,0,
                           0,0,0,0,0,1], 
                 H_matrix=[1,0,0,0,0,0,
                           0,1,0,0,0,0,
                           0,0,1,0,0,0], 
                 P_sigma=150.0, 
                 P_matrix=[0.1,0,0,0,0,0,
                           0,0.1,0,0,0,0,
                           0,0,0.1,0,0,0,
                           0,0,0,1,0,0,
                           0,0,0,0,1,0,
                           0,0,0,0,0,1], 
                 G_sigma=150, 
                 G_matrix=[1,0,0,
                           0,1,0,
                           0,0,1], 
                R_sigma=5.0,
                R_matrix=[1,0,0,
                          0,1,0,
                          0,0,1],
                hypotheses=["P_FP", "P_init", "P_term", "P_link"], 
                lambda_time=5.0, 
                lambda_dist=3.0, 
                lambda_link=1.0, 
                eta=1e-10, 
                theta_dist=20.0,
                theta_time=5.0, 
                dist_thresh=40, 
                time_thresh=2,
                segmentation_miss_rate=0.1,
                relax=True, 
                **kwargs
                 ):
    """
    Build configuration file for btrack with specified parameters
    """
    config = {
        'TrackerConfig': {
            'MotionModel' : {
                'name' : 'platelet_motion', 
                'dt' : dt, 
                "measurements": measurements,
                "states": states,
                "accuracy": accuracy,
                "prob_not_assign": prob_not_assign,
                "max_lost": max_lost,
                'A' : {'matrix' : A_matrix},
                'H' : {'matrix' : H_matrix},
                'P' : {'sigma' : P_sigma, 'matrix' : P_matrix}, 
                'G' : {'sigma' : G_sigma, 'matrix' : G_matrix}, 
                'R' : {'sigma' : R_sigma, 'matrix' : R_matrix}
            }, 
            'ObjectModel' : {},
            'HypothesisModel' : {
                'name' : 'platelet_hypotheses', 
                "hypotheses": hypotheses,
                "lambda_time": lambda_time,
                "lambda_dist": lambda_dist,
                "lambda_link": lambda_link,
                "eta": eta, 
                "theta_dist": theta_dist,
                "theta_time": theta_time,
                "dist_thresh": dist_thresh,
                "time_thresh": time_thresh,
                "segmentation_miss_rate": segmentation_miss_rate,
                "relax": relax
            }
        }
    }
    # save as json
    with open('config-for-optuna.json', 'w') as outfile:
        json.dump(config, outfile)


# -----------------------------------------------------------------------------
# Objective Function
# ------------------

# A class can be used in order to add arguments to the objective function
# maybe this was better done using curry? 
# Oh well, this is how it is done in the optuna example script.

class Objective:
    def __init__(self, df, arr, shape, params, **kwargs):
        """
        Optimise parameters as specified by a params list of form:
        [{'name' : < e.g., 
          'type' : < e.g., 'categorical' > ,
          'options' : < e.g., (<start>, <stop>) ,
          'log' : < e.g., True > ,
          'step' : < e.g., 2
          }]
          Where the name is the name of a parameter in build config, 
            the type is the data type --> trial.suggest_... method,
            options refer to the range or choices of parameters, 
            and log and step are for determining the choices 
        """
        self.df = df   
        self.arr = arr
        self.shape = shape
        self.params = params 

    def __call__(self, trial):
        df = self.df 
        arr = self.arr 
        shape =  self.shape 
        params = self.params 
        param_dict = {}
        for param in params:
            n = param['name']
            t = param['type']
            expr_0 = f'{n} = trial.suggest_{t}(\'{n}\', '
            o = param['options']
            # if categorical, add choices parameter
            if t =='categorical':
                expr_1 = f'choices={o})'
            # if float or int add appropriate parameters
            if t == 'float' or t == 'int':
                expr_1 = f'{o[0]}, {o[1]}' 
                if param.get('log') is not None:
                    l = param['log']
                    expr_1 = expr_1 + f', log={l}'
                if param.get('step') is not None:
                    s = param['step']
                    expr_1 = expr_1 + f', step={s}'
            expr = expr_0 + expr_1 + ')'
            print(expr)
            exec(expr)
            # now that the trial suggest object has been defined
            #    add it to the kwargs for build config
            to_kwargs = f'param_dict[\'{n}\'] = {n}'
            exec(to_kwargs)
        
        # Build Config 
        # ------------
        build_config(**param_dict)
        
        # Tracking
        # --------
        result = _tracking(df, shape)

        # Cost Evaluation
        # ---------------
        if result is not None:
            cost_eval = CostEvaluation(result, arr, shape)
            cost_eval.annotate_all()
            cost = cost_eval.cost
        else:
            cost = 1
        return cost

  
@timeout
def _tracking(df, shape):
    """
    Wrap up the tracking for use with timeout function
    
    Parameters
    ----------
    q: multiprocessing.Queue
    """
    # Tracking
    # --------
    tracks_df = track_objects(
                              df, 
                              shape, 
                              config_name='config-for-optuna.json', 
                              max_search_radius=25
                              )
    return tracks_dfs


# -----------------------------------------------------------------------------
# Execution
# ---------

if __name__ == "__main__":

    # Parser
    # ------
    parser = custom_parser(coords=True, tracks=True, save=True, 
                            base=track_view_base)
    args = parser.parse_args()
    paths = get_paths(args, # will simplify all of this in another PR 
                      'random_tracks_engine',
                      get={'data_path':'image', 
                           'tracks_path':'track',
                           'save_dir':'save',
                           'coords_path': 'coords' 
                           })

    # Read Data
    # ---------
    array = single_zarr(paths['data_path'])
    shape = array.shape
    # tracks = pd.read_csv(paths['tracks_path'])
    coords = pd.read_csv(paths['coords_path'])

    # Cost evaluation attempt
    # -----------------------
    # cost_eval = CostEvaluation(tracks, array, shape)
    # cost_eval.annotate_all()

    # Optimisation Attempt
    # --------------------
    params = [{
        'name' : 'dist_thresh', 
        'type' : 'int',
        'options' : (40, 60), 
        'step' : 10
    }]
    objective = Objective(coords, array, shape, params)
    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

