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

def generate_df_summary(df):
    """Accepts a pandas dataframe and prints out basic details about the data and dataframe structure."""
    
    object_columns = [col for col in df.columns if df[col].dtype == 'object']
    non_object_columns = [col for col in df.columns if df[col].dtype != 'object']
    
    print(f'Dataframe: {df.name}\n')
    print(f'------ Head: ------')
    display(df.head())
    print('\n')
    
    print(f'------ Tail: ------')
    display(df.tail())
    print('\n')
    
    print('------ Column Summaries: ------')
    display(df[object_columns].describe(include='all').transpose())
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
