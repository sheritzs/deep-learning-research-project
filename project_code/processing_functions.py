from IPython.display import display
from darts import concatenate
from darts import TimeSeries
import glob
import json
import numpy as np
import optuna
import os
import pandas as pd
import re
import time
import urllib.request
import warnings

from darts.dataprocessing.transformers import Scaler
from darts.models import (BlockRNNModel, ExponentialSmoothing, LightGBMModel, NBEATSModel,
                          NHiTSModel, RandomForest, XGBModel)
from darts.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveMovingAverage,  NaiveSeasonal
from darts.utils.callbacks import TFMProgressBar
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr
from darts.utils.utils import ModelMode, SeasonalityMode
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import torch
from tqdm.notebook import tqdm

# metrics
from darts.metrics import mae, rmse

non_ml_models = ['ets', 'naive_drift', 'naive_mean', 'naive_moving_average', 'naive_seasonal']


def download_data(api_call: str, file_path: str, file_name: str):
    """
    Accepts an API call and downloads the data under the given file name at the file
    path location. 
    """
    try:
        response = urllib.request.urlopen(api_call)
        data = response.read()

        # decode data
        json_data = json.loads(data.decode('utf-8'))

        # save data to file
        with open(f'{file_path}{file_name}', 'w') as file:
          json.dump(json_data, file)
        print(f'Data successfully downloaded to {file_path}{file_name}.')
        
    except:
        print('Error: file not downloaded')

  
def df_from_json(file):
    """Reads in json weather data and returns a Pandas DataFrame."""
    with open(file) as f:
        contents = f.read()

    json_object = json.loads(contents)
    data = json_object['hourly']

    return pd.DataFrame(data)

def generate_df_summary(df: pd.DataFrame, name:str = None, describe_only: bool = False):
    """Accepts a pandas dataframe and prints out basic details about the data and dataframe structure."""
    
    object_columns = [col for col in df.columns if df[col].dtype == 'object']
    non_object_columns = [col for col in df.columns if df[col].dtype != 'object']
    
    if name:
        print(f'Dataframe: {df.name}\n')

    if describe_only:
        print('------ Column Summaries: ------')
        if object_columns:
            display(df[object_columns].describe(include='all').transpose())
        if non_object_columns:
            display(df[non_object_columns].describe().transpose())
        print('\n')

    else:
        print(f'------ Head: ------')
        display(df.head())
        print('\n')
    
        print(f'------ Tail: ------')
        display(df.tail())
        print('\n')
        
        print('------ Column Summaries: ------')
        if object_columns:
            display(df[object_columns].describe(include='all').transpose())
        if non_object_columns:
            display(df[non_object_columns].describe().transpose())
        print('\n')

        print('------ Counts: ------\n')
        print(f'Rows: {df.shape[0]:,}') 
        print(f'Columns: {df.shape[1]:,}') 
        print(f'Duplicate Rows = {df.duplicated().sum()} | % of Total Rows = {df.duplicated().sum()/df.shape[0]:.1%}') 
        print('\n')

        print('------ Info: ------\n')
        display(df.info()) 
        print('\n')
        
        print('------ Missing Data Percentage: ------')
        display(df.isnull().sum()/len(df) * 100)   

def daily_aggregations(dataframe: pd.DataFrame, convert_time: bool = True) -> pd.DataFrame:
    """Aggregates the weather data at a daily level of granularity."""
    
    df_copy = dataframe.copy()

    mapper = {
        'temperature_2m' : 'temp',
        'relative_humidity_2m' : 'humidity',
    }

    df_copy.rename(columns=mapper, inplace=True)

    if convert_time:
        df_copy['time'] = pd.to_datetime(df_copy['time'])

    df_copy['date'] = df_copy['time'].dt.normalize()
    df_copy = df_copy.set_index('date')
    
    stat_by_variable = {
        'sunshine_duration': 'sum',
        'humidity': 'mean',
    }
    
    # aggregate at the daily level 
    daily_data = df_copy.resample('D').agg(stat_by_variable) 
    df_temp = df_copy['temp'].resample('D').agg([np.min, np.mean, np.max])
    df_temp.columns = [f'temp_{col}' for col in df_temp.columns]
    df_temp['temp_range'] = df_temp['temp_max'] - df_temp['temp_min']
        
    # convert to hourly values 
    daily_data['sunshine_duration'] = daily_data['sunshine_duration'] / 3600 
    daily_data.rename(columns={'sunshine_duration': 'sunshine_hr',
                               'humidity': 'humidity_mean'}, inplace=True)
    
    daily_data = pd.merge(daily_data, df_temp, left_index=True, right_index=True)

    # reorder the columns to display sunshine_hr first
    reordered_columns = ['sunshine_hr'] + [col for col in daily_data if col != 'sunshine_hr']
    
    return daily_data[reordered_columns]

def adjust_outliers(data, columns, granularity='month'):
    """Caps outliers at +/- IQR*1.5 on the specified per-month or per-season basis."""
    
    df_clean = data.copy()
    global_outlier_count = 0
    
    
    for col in columns:
        outlier_count = 0
    
        if granularity == 'month':
            for month in set(df_clean['month'].unique()):
                Q1 = df_clean[df_clean['month'] == month][col].quantile(0.25)
                Q3 = df_clean[df_clean['month'] == month][col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5*IQR
                upper = Q3 + 1.5*IQR

                outlier_count +=  len(df_clean[(df_clean['month'] == month) & (df_clean[col] > upper)]) + \
                len(df_clean[(df_clean['month'] == month) & (df_clean[col] < lower)])


                if outlier_count > 0:
                    df_clean[col] = np.where((df_clean['month'] == month) & (df_clean[col] > upper), upper, df_clean[col])
                    df_clean[col] = np.where((df_clean['month'] == month) & (df_clean[col] < lower), lower, df_clean[col])

        elif granularity == 'season':
            for season in set(df_clean['season_str'].unique()):
                Q1 = df_clean[df_clean['season_str'] == season][col].quantile(0.25)
                Q3 = df_clean[df_clean['season_str'] == season][col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5*IQR
                upper = Q3 + 1.5*IQR


                outlier_count +=  len(df_clean[(df_clean['season_str'] == season) & (df_clean[col] > upper)]) + \
                len(df_clean[(df_clean['season_str'] == season) & (df_clean[col] < lower)])

                if outlier_count > 0:
                    df_clean[col] = np.where((df_clean['season_str'] == season) & (df_clean[col] > upper), upper, df_clean[col])
                    df_clean[col] = np.where((df_clean['season_str'] == season) & (df_clean[col] < lower), lower, df_clean[col])

        else:
            print('Invalid granularity specified. Please indicate "month" or "season".')
            return None


        global_outlier_count += outlier_count
        print(f'Total outliers adjusted in the {col} column: {outlier_count:,}')
        print(f'Percent of total rows: {outlier_count/len(df_clean):.2%}')
        print('\n')

    return df_clean

def create_timeseries(df, col):
  """Creates a TimeSeries object for the given column with data type float 32 for quicker training/processing."""
  df = df.copy().reset_index()
  return TimeSeries.from_dataframe(df[['date', col]], 'date', col).astype(np.float32) 

def get_covariate_ts(df):
    """Returns timeseries objects for the combined covariates. """
    df = df.copy().reset_index()
    
    time_series = {
        'covariates': {}
        }
    
    for col in df.columns[2:]:
        time_series['covariates'][col] = create_timeseries(df, col)

    # create stacked timeseries for the covariates (exogenous variables)
    covariates_ts = concatenate([ts for ts in time_series['covariates'].values()],
                              axis=1)

    return covariates_ts

def get_clean_df(df, agg_cols):
    """Aggregates data and removes outliers on a per-month basis. """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    
    # create daily aggregations 
    df = daily_aggregations(df, agg_cols)
    df.drop(['humidity_min', 'humidity_max'], axis=1, inplace=True)
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['month'] = df.index.month
    
    month_label_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
               'Sep', 'Oct', 'Nov', 'Dec']
    
    cols_to_adjust = list(df.columns[:-1])
    
    df_clean = adjust_outliers(df, columns=cols_to_adjust, granularity='month')
    df_clean.drop('month', axis=1, inplace=True)

    return df_clean
    
def post_results(results, file, mode='a', create_backup=False):
    """Records results to a .json file, with an optional backup."""

    try:
        with open(file, mode) as output_file:
            json.dump(results, output_file)
            print(f'\nSuccessfully posted results to {file}')

        # create backup file 
        if create_backup:
            split_arr = file.split('.')
            backup_file = f'{split_arr[0]}_backup.{split_arr[1]}'
            backup_file

            with open(backup_file, mode) as output_file:
                json.dump(results, output_file)
                print(f'Successfully posted results to {backup_file}\n')

    except Exception as e:
        print('Unable to save results to file')
        print(e)

def read_json_file(file, output_type='dict'):
    """Reads in json file and returns a dictionary or pandas dataframe."""

    with open(file) as json_file:
        if output_type == 'dict':
            data = json.load(json_file)
        elif output_type == 'df':
            data = pd.read_json(json_file)

    return data 

def print_callback(study, trial):
  print(f"Current value: {trial.value}, Current params: {trial.params}")
  print(f"Current Best value: {study.best_value}, Best params: {study.best_trial.params}")

def get_model(model_name, fh, hyperparams, seed, version=None,
              model_type='default', n_epochs_override=None):

    """Returns an unfitted model and a semi-unique moniker based on the given arguments, including model version in the case of N-BEATS."""

    if model_name == 'nbeats': 
        model_name_fh = f'{model_name}_{model_type}_{version}_fh{fh}' 
    elif model_name not in non_ml_models:
        model_name_fh = f'{model_name}_{model_type}_fh{fh}'
    else:
        model_name_fh = f'{model_name}_fh{fh}'

    if model_name in non_ml_models:
        if model_name == 'ets':
            model = ExponentialSmoothing(trend=ModelMode.ADDITIVE,
                                        seasonal=SeasonalityMode.ADDITIVE,
                                        seasonal_periods=365)    
        if model_name == 'naive_drift':
            model = NaiveDrift()
        if model_name == 'naive_seasonal':
            model = NaiveSeasonal(K=365)
        if model_name == 'naive_mean':
            model = NaiveMean()
        if model_name == 'naive_moving_average':
            model = NaiveMovingAverage(input_chunk_length=fh*2)
        
        return model, model_name_fh, n_epochs_override

    if model_name in ['lstm', 'gru', 'nbeats', 'nhits']:

        # detect whether a GPU is available
        if torch.cuda.is_available():
            pl_trainer_kwargs = {
                'accelerator': 'gpu'
            }
        else:
            pl_trainer_kwargs = None

        
    hyp = hyperparams[model_name] 

    if model_type == 'default':

        if model_name in ['lstm', 'gru']:

            if n_epochs_override:
                model = BlockRNNModel(
                    model = model_name.upper(),
                    input_chunk_length = fh * 2,
                    output_chunk_length = fh,
                    n_epochs = n_epochs_override,
                    pl_trainer_kwargs = pl_trainer_kwargs,
                )

            else:
                model = BlockRNNModel(
                    model = model_name.upper(),
                    input_chunk_length = fh * 2,
                    output_chunk_length = fh,
                    pl_trainer_kwargs = pl_trainer_kwargs,
                )

        elif model_name == 'nbeats':

            if n_epochs_override:
                model = NBEATSModel(
                    input_chunk_length = fh * 2,
                    output_chunk_length = fh,
                    generic_architecture = True if version == 'generic' else False,
                    n_epochs = n_epochs_override,
                    pl_trainer_kwargs = pl_trainer_kwargs
                )
            else:
                model = NBEATSModel(
                    input_chunk_length = fh * 2,
                    output_chunk_length = fh,
                    generic_architecture = True if version == 'generic' else False,
                    pl_trainer_kwargs = pl_trainer_kwargs
                )

        elif model_name == 'nhits':

            if n_epochs_override:
                model = NHiTSModel(
                    input_chunk_length = fh * 2,
                    output_chunk_length = fh,
                    n_epochs = n_epochs_override,
                    pl_trainer_kwargs = pl_trainer_kwargs
                )
            else:
                model = NHiTSModel(
                    input_chunk_length = fh * 2,
                    output_chunk_length = fh,
                    pl_trainer_kwargs = pl_trainer_kwargs
                )

        elif model_name == 'rf':
            model = RandomForest(
                lags = fh*2,
                lags_past_covariates = fh*2,
                output_chunk_length = fh
            )

        elif model_name == 'xgboost':
            model = XGBModel(
                lags = fh*2,
                lags_past_covariates = fh*2,
                output_chunk_length = fh,
                random_state=seed
            )

        elif model_name == 'lgbm':
            model = LightGBMModel(
                lags = fh*2,
                lags_past_covariates = fh*2,
                output_chunk_length = fh,
                verbose=-1,
                random_state=seed
            )

    elif model_type == 'tuned':

        if model_name in ['lstm', 'gru']:

            model = BlockRNNModel(
                model = model_name.upper(),
                input_chunk_length = hyp[fh]['parameters']['input_chunk_length'],
                output_chunk_length = fh,
                batch_size =  hyp[fh]['parameters']['batch_size'],
                n_epochs = hyp[fh]['parameters']['n_epochs'] if n_epochs_override is None else n_epochs_override, 
                hidden_dim = hyp[fh]['parameters']['hidden_dim'],
                n_rnn_layers = hyp[fh]['parameters']['n_rnn_layers'],
                dropout = round(hyp[fh]['parameters']['dropout'],7),
                pl_trainer_kwargs = pl_trainer_kwargs,
                optimizer_kwargs = {'lr': round(hyp[fh]['parameters']['lr'],7) },
            )

        elif model_name == 'nbeats':

            model = NBEATSModel(
                random_state=seed,
                input_chunk_length = hyp[version][fh]['parameters']['input_chunk_length'],
                output_chunk_length = fh,
                num_stacks = hyp[version][fh]['parameters']['num_stacks'],
                num_blocks = hyp[version][fh]['parameters']['num_blocks'],
                num_layers = hyp[version][fh]['parameters']['num_layers'],
                layer_widths = hyp[version][fh]['parameters']['layer_widths'],
                batch_size = hyp[version][fh]['parameters']['batch_size'],
                n_epochs = hyp[version][fh]['parameters']['n_epochs'] if n_epochs_override is None else n_epochs_override,
                dropout = round(hyp[version][fh]['parameters']['dropout'],7),
                activation =  hyp[version][fh]['parameters']['activation'],
                generic_architecture=True if version == 'generic' else False,
                pl_trainer_kwargs = pl_trainer_kwargs,
                optimizer_kwargs = {'lr': round(hyp[version][fh]['parameters']['lr'],7) },
            )

        elif model_name == 'nhits':

            model = NHiTSModel(
                random_state=seed,
                input_chunk_length = hyp[fh]['parameters']['input_chunk_length'],
                output_chunk_length = fh,
                num_stacks = hyp[fh]['parameters']['num_stacks'],
                num_blocks = hyp[fh]['parameters']['num_blocks'],
                num_layers = hyp[fh]['parameters']['num_layers'],
                layer_widths = hyp[fh]['parameters']['layer_widths'],
                batch_size = hyp[fh]['parameters']['batch_size'],
                n_epochs = hyp[fh]['parameters']['n_epochs'] if n_epochs_override is None else n_epochs_override,
                dropout = round(hyp[fh]['parameters']['dropout'],7),
                pl_trainer_kwargs = pl_trainer_kwargs,
                optimizer_kwargs = {'lr': round(hyp[fh]['parameters']['lr'],7) },
            )

        elif model_name == 'rf':
            model = RandomForest(
                lags = hyp[fh]['parameters']['lags'],
                lags_past_covariates = hyp[fh]['parameters']['lags_past_covariates'],
                n_estimators = hyp[fh]['parameters']['n_estimators'],
                max_depth = hyp[fh]['parameters']['max_depth'],
                output_chunk_length = fh,
                random_state=seed
            )

        elif model_name == 'xgboost':
            model = XGBModel(
                lags = hyp[fh]['parameters']['lags'],
                lags_past_covariates = hyp[fh]['parameters']['lags_past_covariates'],
                output_chunk_length = fh,
                random_state=seed
            )

        elif model_name == 'lgbm':
            model = LightGBMModel(
                lags = hyp[fh]['parameters']['lags'],
                lags_past_covariates = hyp[fh]['parameters']['lags_past_covariates'],
                output_chunk_length = fh,
                verbose=-1,
                random_state=seed
            )

    return model, model_name_fh, n_epochs_override

def get_reformatted_hyperparams(hyp_dict, forecast_horizons):

    """
    Accepts a dictionary with the results of hyperparameter tuning and returns a reformatted version
    for the machine learning experiments.
    """

    new_hyp_dict = {

        'gru': {fh: {} for fh in forecast_horizons},
        'lgbm': {fh: {} for fh in forecast_horizons},
        'lstm': {fh: {} for fh in forecast_horizons},
        'nbeats': {
            'generic': {fh: {} for fh in forecast_horizons},
            'interpretable': {fh: {} for fh in forecast_horizons}
        },
        'nhits': {fh: {} for fh in forecast_horizons},
        'rf': {fh: {} for fh in forecast_horizons},
        'xgboost': {fh: {} for fh in forecast_horizons}
    }

    for hyp_name, values in hyp_dict.items():

        if len(hyp_name.split('_')) == 4: # nbeats's name has 4 components, incl. version
            
            _, model_name_main, version, fh_str = hyp_name.split('_')
            fh = int(re.search(r'(?:h)(.*)', fh_str).group(1))

            new_hyp_dict[model_name_main][version][fh] = {
                'parameters': values['best_parameters'],
                'training_rmse': values['best_rmse'],
                'hyp_search_time': values['hyperparam_search_time']
            }

        else:

            _, model_name_main, fh_str = hyp_name.split('_')
            fh = int(re.search(r'(?:h)(.*)', fh_str).group(1))

            new_hyp_dict[model_name_main][fh] = {
                'parameters': values['best_parameters'],
                'training_rmse': values['best_rmse'],
                'hyp_search_time': values['hyperparam_search_time']
            }

    return new_hyp_dict

def run_experiment(model, model_names, n_epochs_override, hyperparameters, cutoff_date, fh, 
                   df_outliers, df_clean, has_outliers, results,
                   models_directory, results_directory, seed=None, verbose=True):
    
    """Runs an experiment and saves the results to a file."""
    current_results = results.copy()

    model_name = model_names[0]
    model_name_proper = model_names[1]
    model_name_fh = model_names[2]

    print(f'\nRunning {model_name_fh} Experiments - Forecast Horizon: {fh} | Outlier Flag: {has_outliers}...\n') 

    target_train, target_test, cov_train = train_test_split(cutoff_date, df_outliers, df_clean,  has_outliers=has_outliers)

    if model_name not in non_ml_models and model_name != 'nbeats':
        target_scaler = Scaler()
        target_train = target_scaler.fit_transform(target_train)
        cov_scaler = Scaler() 
        cov_train = cov_scaler.fit_transform(cov_train)

    start_time = time.perf_counter()

    if model_name in non_ml_models:
        model.fit(series=target_train)

    elif model_name in ['nbeats', 'lstm', 'gru', 'nhits']:
        if seed:
            torch.manual_seed(seed)
        if model_name == 'nbeats':
            model.fit(series=target_train,
                        past_covariates=cov_train,
                        verbose=verbose)
        else:
            model.fit(series=target_train,
                        past_covariates=cov_train)
            
        model.save(f'{models_directory}{model_name_fh}_fitted.pt') 

    else:
        model.fit(series=target_train,
                    past_covariates=cov_train)
        model.save(f'{models_directory}{model_name_fh}_fitted.pkl')

    end_time = time.perf_counter()
    training_time = round((end_time - start_time) / 60, 3)

    if model_name in non_ml_models:
        predictions = model.predict(n=fh)
    else:
        predictions = model.predict(n=fh,
                                    series=target_train,
                                    past_covariates=cov_train)
        
    if model_name not in non_ml_models and model_name != 'nbeats':
        predictions = target_scaler.inverse_transform(predictions)

    rmse_score = round(rmse(predictions, target_test[:fh]), 4)
    mae_score = round(mae(predictions, target_test[:fh]), 4)

    key =  f"optuna_{model_name_fh.replace('_default', '').replace('_tuned', '')}"

    if 'tuned' in model_name_fh:
        hyp_search_time = round(hyperparameters[key]['hyperparam_search_time'], 3)
        best_val_rmse = round(hyperparameters[key]['best_rmse'], 4)
        total_time = training_time + hyp_search_time 

        if model_name in ['nbeats', 'nhits', 'lstm', 'gru']:
            if n_epochs_override is None:
                n_epochs = hyperparameters[key]['best_parameters']['n_epochs']
            else:
                n_epochs = n_epochs_override
        else:
            n_epochs = np.nan

    else:
        hyp_search_time = np.nan
        best_val_rmse = np.nan
        total_time = round(training_time, 3)

    if model_name in non_ml_models:
        model_type = 'default'
    else:
        model_type = model_name_fh.split('_')[1]

    if model_type == 'default' and n_epochs_override is None and model_name in ['nbeats', 'nhits', 'lstm', 'gru']:
        n_epochs = 100
    elif n_epochs_override:
        if model_name not in ['nbeats', 'nhits', 'lstm', 'gru']:
            n_epochs = np.nan
        else:
            n_epochs = n_epochs_override
    else:
        try:
            n_epochs
        except NameError:
            n_epochs = np.nan


    has_n_epochs_override = True if n_epochs_override else False


    # Record results
    current_results['model_name_proper'].append(model_name_proper) 
    current_results['model_name_fh'].append(model_name_fh)
    current_results['model_type'].append(model_type)
    current_results['has_outliers'].append(has_outliers)
    current_results['forecast_horizon'].append(fh)
    current_results['rmse'].append(rmse_score)
    current_results['mae'].append(mae_score)
    current_results['n_epochs'].append(n_epochs)
    current_results['has_n_epochs_override'].append(has_n_epochs_override)
    current_results['training_time'].append(training_time)
    current_results['hyp_search_time'].append(hyp_search_time)
    current_results['best_val_rmse'].append(best_val_rmse)
    current_results['total_time'].append(total_time)

    results.update(current_results) 
    
    # if model_name == 'nbeats': # breaking up the N-BEATS experiments to avoid Colab execution timeout and progress/data loss
    #     if model_type == 'default':
    #         file_name = f'{results_directory}{model_name}_{model_type}_outliers-{has_outliers}_epoch-override-{has_n_epochs_override}_results.csv'
    #     else:
    #         file_name = f'{results_directory}{model_name}_{model_type}_epoch-override-{has_n_epochs_override}_results.csv'
    # else:
    #     file_name = f'{results_directory}{model_name}_results.csv'

    path = f'{results_directory}cutoff_date={cutoff_date}/'

    if not os.path.exists(path):
        os.makedirs(path)

    file_name = f'{path}{model_name}_cutoffdate={cutoff_date}_results.csv' 
    pd.DataFrame(results).to_csv(file_name, index=False)

def train_test_split(cutoff_date, df_outliers=None, df_clean=None, has_outliers=False):

    if has_outliers==False:

        target = create_timeseries(df_clean, 'sunshine_hr')
        # create past covariates as stacked timeseries of exogenous variables
        past_cov = get_covariate_ts(df_clean)

    elif has_outliers == True:

        target = create_timeseries(df_outliers, 'sunshine_hr')
        past_cov = get_covariate_ts(df_outliers)

    # create training and testing datasets
    training_cutoff = pd.Timestamp(cutoff_date)

    target_train, target_test = target.split_after(training_cutoff)
    cov_train, _ = past_cov.split_after(training_cutoff)

    return target_train, target_test, cov_train 


def highlight_min_max(df:pd.DataFrame, columns_to_drop:list=None, index_col='model_name', highlight_selection:str='all', print_latex=True):
    """"Highlights the minimum or maximum value in each row within a given df."""

    if columns_to_drop is None:
        df_copy = df.copy()
    else:
        df_copy = df.copy().drop(columns_to_drop, axis=1)

    df_copy.set_index(index_col, inplace=True)
    df_styled = df_copy.style.format("{:.3f}")
    
    if highlight_selection == 'all':
        df_styled.highlight_min(axis=0, props='font-weight:bold; color:darkgreen;').highlight_max(axis=0, props='font-weight:bold; color:firebrick;')

        if print_latex:
            print('Latex Version: \n')
            print(df_styled.to_latex(convert_css=True))

        df_styled.highlight_min(color='mediumseagreen').highlight_max(color='tomato')

    elif highlight_selection == 'min':
        df_styled.highlight_min(axis=0, props='font-weight:bold; color:darkgreen;')

        if print_latex:
            print('Latex Version: \n')
            print(df_styled.to_latex(convert_css=True))

        df_styled.highlight_min(color='mediumseagreen')

    elif highlight_selection == 'max':
        df_styled.highlight_max(axis=0, props='font-weight:bold; color:firebrick;')

        if print_latex:
            print('Latex Version: \n')
            print(df_styled.to_latex(convert_css=True))
            
        df_styled.highlight_max(color='tomato') 

    return df_styled

class PyTorchLightningPruningCallback(Callback):
    """
    PyTorch Lightning callback to prune unpromising trials
    and address minor issue due to PyTorch-lighting default sanity check value.
    source: https://github.com/optuna/optuna-examples/issues/166#issuecomment-1403112861

    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

def print_callback(study, trial):
  """Optional callback for sanity checks during Optuna trials."""
  print(f"Current value: {trial.value}, Current params: {trial.params}")
  print(f"Current Best value: {study.best_value}, Best params: {study.best_trial.params}")

def generate_cutoff_date(start_date: str, end_date:str, seed=None, n=1, replace=False) -> str | list:
    """
    Generates one or more random dates from a given range, i.e. start_date to end_date (inclusive)
    Entry date format: 'yyyy-mm-dd'; e.g. '2000-01-01'
    """
    all_dates = pd.date_range(start_date, end_date).to_series()
    selected_dates = all_dates.sample(n, replace=replace, random_state=seed)
    final_dates = [date.strftime('%Y-%m-%d') for date in selected_dates]

    if n == 1:
        return final_dates[0]
    else:
        return final_dates

def generate_error_table(df:pd.DataFrame, required_columns:list, index:list, 
                          pivot_column='FH', error_metric='rmse', outlier_split=True):
    """Generates a summary table for the given error metric. """ 
    required_columns = required_columns + [error_metric]

    error_table = df[required_columns]\
                        .pivot_table(index=index, columns=pivot_column, values=error_metric)\
                        .loc[:, ['FH-1', 'FH-3', 'FH-7', 'FH-14', 'FH-28']]\
                        .sort_values(by=['has_outliers', 'model_name'], ascending=[False, True])\
                        .reset_index()
    
    error_table['Median'] = error_table.iloc[:, 2:].median(axis=1)
    error_table['Mean'] = round(error_table.iloc[:, 2:].mean(axis=1),4)
    
    if outlier_split:
        error_table_outliers = error_table[error_table['has_outliers'] == True]
        error_table_no_outliers = error_table[error_table['has_outliers'] == False]
        return error_table_outliers, error_table_no_outliers 
    else:
        return error_table
