import pandas as pd
import os

def load_data(file_name):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data', 'raw')
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} was not found in the raw data directory.")
    
    return pd.read_csv(file_path)