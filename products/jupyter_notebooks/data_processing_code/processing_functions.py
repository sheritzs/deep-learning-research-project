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
