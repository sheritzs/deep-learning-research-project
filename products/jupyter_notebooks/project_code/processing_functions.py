from IPython.display import display
import json
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
