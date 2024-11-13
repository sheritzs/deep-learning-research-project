import warnings
warnings.filterwarnings(
    "ignore"
)

import logging
logging.disable(logging.CRITICAL)

import datetime
from IPython.display import display
import json
import numpy as np
import pandas as pd
from project_code import processing_functions as pf
import time


from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
from darts.models import (BlockRNNModel, LightGBMModel, NBEATSModel, RandomForest, XGBModel)
from darts.utils.callbacks import TFMProgressBar
import kaleido
from pytorch_lightning.callbacks import Callback
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_intermediate_values 
from optuna.visualization import plot_optimization_history
import torch



def objective_nbeats(trial: optuna.Trial, common_inputs:dict,  version: str, fh: int, 
                  model_name_fh: int, seed: int) -> float:
    
    """Hyperparameter search objective"""

    pruner = pf.PyTorchLightningPruningCallback(trial, monitor='val_loss')
    callbacks = [pruner]

    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            'accelerator': 'gpu',
            'callbacks': callbacks,
        }
    else:
        pl_trainer_kwargs = {'callbacks': callbacks}

    input_chunk_lengths = common_inputs['input_chunk_lengths']
    batch_sizes = common_inputs['batch_sizes']

    nbeats_params = {
                    'input_chunk_length': trial.suggest_categorical('input_chunk_length', input_chunk_lengths),
                    'output_chunk_length': fh, 
                    'batch_size': trial.suggest_categorical('batch_size', batch_sizes),
                    'num_stacks': trial.suggest_categorical('num_stacks', [10, 20, 30]), # only used in model if generic_architecture is set to True,
                    'num_blocks': trial.suggest_categorical('num_blocks', [1, 2, 3]),
                    'num_layers': trial.suggest_categorical('num_layers', [3, 4, 5]),
                    'layer_widths': trial.suggest_categorical('layer_widths', [256, 512]),
                    'dropout': trial.suggest_float('dropout', 0, 0.4),
                    'optimizer_kwargs': {'lr': trial.suggest_float('lr',  1e-5, 1e-1, log=True)},
                    'n_epochs': trial.suggest_int('n_epochs', 15, 150, step=15),
                    'activation': trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU']),
                    'generic_architecture': True if version == 'generic' else False,
                    'pl_trainer_kwargs': pl_trainer_kwargs,
                    'model_name': f'{model_name_fh}_{datetime.datetime.now().strftime(("%Y%m%d-%H%M%S"))}',
                    'save_checkpoints': True,
                    'force_reset': True,
                    'random_state': seed
                    }

    model = NBEATSModel(**nbeats_params)

    model.fit(
        series=common_inputs['unscaled_data']['target_train'],
        past_covariates=common_inputs['unscaled_data']['cov_train'],
        )

    predictions = model.predict(n=fh,
                                series=common_inputs['unscaled_data']['target_train'],
                                past_covariates=common_inputs['unscaled_data']['cov_train']
                                )
    rmse_score = rmse(predictions, common_inputs['target_test'][:fh])

    return rmse_score

def objective_rnn(trial: optuna.Trial,  common_inputs:dict,  version: str, fh: int, 
                  model_name_fh: int, seed: int ) -> float: 

    """Hyperparameter search objective"""

    pruner = pf.PyTorchLightningPruningCallback(trial, monitor='val_loss')
    callbacks = [pruner]

    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            'accelerator': 'gpu',
            'callbacks': callbacks,
        }
    else:
        pl_trainer_kwargs = {'callbacks': callbacks}

    input_chunk_lengths = common_inputs['input_chunk_lengths']
    batch_sizes = common_inputs['batch_sizes']

    rnn_params = {
                    'input_chunk_length': trial.suggest_categorical('input_chunk_length', input_chunk_lengths),
                    'output_chunk_length': fh, 
                    'batch_size': trial.suggest_categorical('batch_size', batch_sizes),
                    'hidden_dim': trial.suggest_int('hidden_dim', 10, 40),
                    'n_rnn_layers': trial.suggest_int('n_rnn_layers', 2, 10),
                    'dropout': trial.suggest_float('dropout', 0, 0.4),
                    'optimizer_kwargs': {'lr': trial.suggest_float('lr',  1e-5, 1e-1, log=True)},
                    'n_epochs': trial.suggest_int('n_epochs', 15, 150, step=15),
                    'model': version,
                    'pl_trainer_kwargs': pl_trainer_kwargs,
                    'model_name': f'{model_name_fh}_{datetime.datetime.now().strftime(("%Y%m%d-%H%M%S"))}',
                    'save_checkpoints': True,
                    'force_reset': True,
                    'random_state': seed
                    }

    model = BlockRNNModel(**rnn_params)

    model.fit(
        series=common_inputs['scaled_data']['target_train'],
        past_covariates=common_inputs['scaled_data']['cov_train'],
        )

    predictions = model.predict(n=fh,
                                series=common_inputs['scaled_data']['target_train'],
                                past_covariates=common_inputs['scaled_data']['cov_train']
                                )
    
    target_scaler = common_inputs['scaled_data']['target_scaler']
    predictions = target_scaler.inverse_transform(predictions)
    rmse_score = rmse(predictions, common_inputs['target_test'][:fh])

    return rmse_score

def objective_rf(trial: optuna.Trial,  common_inputs:dict, fh: int, 
                  model_name_fh: int, seed: int) -> float: 

    LAGS = common_inputs['input_chunk_lengths']
    N_ESTIMATORS = [50, 100, 150, 200]
    MAX_DEPTH = [2, 5, 10, 15, 20]

    rf_params = {
                    'lags': trial.suggest_categorical("lags", LAGS),
                    'lags_past_covariates': trial.suggest_categorical('lags_past_covariates', LAGS),
                    'n_estimators': trial.suggest_categorical('n_estimators', N_ESTIMATORS),
                    'max_depth': trial.suggest_categorical('max_depth',  MAX_DEPTH),
                    'output_chunk_length': fh
                    }

    model = RandomForest(**rf_params)

    model.fit(
        series=common_inputs['scaled_data']['target_train'],
        past_covariates=common_inputs['scaled_data']['cov_train']
        )

    predictions = model.predict(n=fh,
                                series=common_inputs['scaled_data']['target_train'],
                                past_covariates=common_inputs['scaled_data']['cov_train']
                                )
    
    target_scaler = common_inputs['scaled_data']['target_scaler']
    predictions = target_scaler.inverse_transform(predictions)
    rmse_score = rmse(predictions, common_inputs['target_test'][:fh])

    return rmse_score

def objective_xgb(trial: optuna.Trial, fh: int, model_name_fh: int, 
                  common_inputs: dict, seed: int ) -> float:
    pass

def objective_lgbm(trial: optuna.Trial, fh: int, model_name_fh: int, 
                  common_inputs: dict, seed: int ) -> float:
    pass


def hyperparameter_search(fh, model_name, common_inputs, n_trials, results_dict,
                          results_directory, hyperparam_file, version=None, seed=None):

    if model_name == 'nbeats':
        model_name_fh = f'optuna_{model_name}_{version}_fh{fh}'
    else:
        model_name_fh = f'optuna_{model_name}_fh{fh}'

    print(f'Running hyperparameter search for {model_name_fh}\n')

    start_time = time.perf_counter()

    study = optuna.create_study(direction='minimize') 

    if model_name in ['lstm', 'gru']:
        version = version.upper()
        func = lambda trial: objective_rnn(trial, common_inputs, version, fh, model_name_fh, seed) 
    elif model_name == 'nbeats':
        func = lambda trial: objective_nbeats(trial, common_inputs, version, fh, model_name_fh, seed)
    elif model_name == 'rf':
        func = lambda trial: objective_rf(trial, common_inputs, fh, model_name_fh, seed)

    study.optimize(func, n_trials=n_trials)
    end_time = time.perf_counter()
    operation_runtime = round((end_time - start_time)/60, 2)

    results = {model_name_fh: {
            'best_rmse': round(study.best_value, 4),
            'best_parameters': study.best_trial.params,
            'hyperparam_search_time': operation_runtime
        }}

    # save trial results
    study.trials_dataframe().to_csv(f'{results_directory}{model_name_fh}_trials.csv')
    results_dict.update(results)
    pf.post_results(results_dict, hyperparam_file, 'w')

    fig = plot_optimization_history(study, target_name='RMSE')
    fig.update_layout(
            title=f'Hyperparameter Optimization History for {model_name_fh}',
            xaxis_title="Trial",
            yaxis_title="RMSE",
            showlegend=True
            )
    fig.write_image(f'{results_directory}figures/{model_name_fh}_trial_history.png')
    
    print(f'\nHyperparameter search for {model_name_fh} completed.\n')
