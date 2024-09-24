from IPython.display import display
from darts import concatenate
from darts import TimeSeries
import json
import numpy as np
import pandas as pd
import urllib.request


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

def generate_df_summary(df, describe_only=False):
    """Accepts a pandas dataframe and prints out basic details about the data and dataframe structure."""
    
    object_columns = [col for col in df.columns if df[col].dtype == 'object']
    non_object_columns = [col for col in df.columns if df[col].dtype != 'object']
    
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


def daily_aggregations(dataframe):
    """Aggregates the weather data at a daily level of granularity."""

    df_copy = dataframe.copy()
    df_copy['date'] = df_copy['time'].dt.normalize()
    df_copy = df_copy.set_index('date')

    daily_data = df_copy.loc[:, ['sunshine_s', 'precipitation', 'shortwave_radiation']].resample('D').sum()

    # convert to hourly values 
    daily_data['sunshine_hr'] = daily_data['sunshine_s'] / 3600

    # columns for min, mean, and max aggregations
    agg_cols = ['temp', 'humidity', 'dew_point', 'cloud_cover', 'wind_speed']

    # Compute min, mean, and max values

    # minimum aggregations
    mins = df_copy[agg_cols].resample('D').min()
    col_names_min = [f'min_{name}' for name in agg_cols]
    mins.columns = col_names_min

    # mean aggregations
    means = df_copy[agg_cols].resample('D').mean()
    col_names_mean = [f'mean_{name}' for name in agg_cols]
    means.columns = col_names_mean

    # max aggregations
    maxes = df_copy[agg_cols].resample('D').max()
    col_names_max = [f'max_{name}' for name in agg_cols]
    maxes.columns = col_names_max

    # merge the aggregated dataframes
    for df in [mins, means, maxes]:
        daily_data = pd.merge(daily_data, df, left_index=True, right_index=True)

    daily_data = daily_data.drop(columns='sunshine_s').round(3)

    # reorder the columns to display sunshine_hr first
    daily_data = daily_data.loc[:, ['sunshine_hr', 'shortwave_radiation', 'precipitation', 
                                    'min_temp','mean_temp', 'max_temp',
                                    'min_humidity', 'mean_humidity', 'max_humidity',
                                    'min_dew_point','mean_dew_point', 'max_dew_point',
                                    'min_cloud_cover',  'mean_cloud_cover', 'max_cloud_cover',
                                    'min_wind_speed', 'mean_wind_speed', 'max_wind_speed']]

    return daily_data

def daily_aggregations_v2(dataframe, agg_cols):
    """Aggregates the weather data at a daily level of granularity."""
    
    df_copy = dataframe.copy()
    df_copy['date'] = df_copy['time'].dt.normalize()
    df_copy = df_copy.set_index('date')
    
    daily_data = df_copy.loc[:, ['sunshine_duration']].resample('D').sum()
    
    # convert to hourly values 
    daily_data['sunshine_hr'] = daily_data['sunshine_duration'] / 3600
    
    # # Compute min, mean, and max values
    
    # # minimum aggregations
    mins = df_copy[agg_cols].resample('D').min()
    col_names_min = [f'{name}_min' for name in agg_cols]
    mins.columns = col_names_min
    
    # # mean aggregations
    means = df_copy[agg_cols].resample('D').mean()
    col_names_mean = [f'{name}_mean' for name in agg_cols]
    means.columns = col_names_mean
    
    # # max aggregations
    maxes = df_copy[agg_cols].resample('D').max()
    col_names_max = [f'{name}_max' for name in agg_cols]
    maxes.columns = col_names_max
    
    # # merge the aggregated dataframes
    for df in [mins, means, maxes]:
        daily_data = pd.merge(daily_data, df, left_index=True, right_index=True)
    
    daily_data = daily_data.drop(columns='sunshine_duration').round(3)
    
    # # reorder the columns to display sunshine_hr first
    reordered_columns = sorted([col for col in daily_data if col != 'sunshine_hr'])
    reordered_columns.insert(0, 'sunshine_hr')
    
    daily_data = daily_data[reordered_columns]

    return daily_data

def get_season(month_day, data_type='string'):
    """
    Returns a season indicator based on a given month_day value.
    There is roughly a 1-3 day margin of error, given the 
    seasonal timeline in any given year.

    month_day: a value calculated based on month of year and day of month,
    such that January 1st = 101 and December 31st = 1231. 

    """
    try:

        if data_type == 'string':

            if ((month_day >= 320) and (month_day <= 619)):
                season = "Spring"
            elif ((month_day >= 620) and (month_day <= 921)):
                season = "Summer"
            elif ((month_day >= 922) and (month_day <= 1219)):
                season = "Fall"
            elif ((month_day >= 1220) or (month_day <= 319)):
                season = "Winter"
            else:
                raise IndexError("Invalid month_day Input")

        elif data_type == 'int':

            if ((month_day >= 320) and (month_day <= 619)):
                season = 1
            elif ((month_day >= 620) and (month_day <= 921)):
                season = 2
            elif ((month_day >= 922) and (month_day <= 1219)):
                season = 3
            elif ((month_day >= 1220) or (month_day <= 319)):
                season = 4
            else:
                raise IndexError("Invalid month_day Input")

        return season

    except:
        error_string = "Error: data_type selected should be 'int' or 'string' "
        return error_string
    


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
    df = df.copy().reset_index()
    
    """Returns timeseries objects for the combined covariates. """
    
    time_series = {
        'covariates': {}
        }
    
    for col in df.columns[2:]:
        time_series['covariates'][col] = create_timeseries(df, col)

    # create stacked timeseries for the covariates
    covariates_ts = concatenate([ts for ts in time_series['covariates'].values()],
                              axis=1)

    return covariates_ts

def get_clean_df(df, agg_cols):
    """Aggregates data and removes outliers on a per-month basis. """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    
    # create daily aggregations 
    df = daily_aggregations_v2(df, agg_cols)
    df.drop(['humidity_min', 'humidity_max'], axis=1, inplace=True)
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['month'] = df.index.month
    
    month_label_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
               'Sep', 'Oct', 'Nov', 'Dec']
    
    cols_to_adjust = list(df.columns[:-1])
    
    df_clean = adjust_outliers(df, columns=cols_to_adjust, granularity='month')
    df_clean.drop('month', axis=1, inplace=True)

    return df_clean
    
def post_hyperparam_results(results, file, mode='a'):
    """Records the best hyper parameter search results to a .json file."""

    try:
        with open(file, mode) as output_file:
            json.dump(results, output_file)
            print(f'Successfully posted results to {file}')
    except Exception as e:
        print('Unable to save results to file')
        print(e)
