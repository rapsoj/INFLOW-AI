import os
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Constants for file URL and paths
URL = 'https://blueice.gsfc.nasa.gov/gwm/timeseries/lake000314.10d.2.txt'
FOLDER_PATH = 'Satellite Altimetry etc'
FILE_PATH = os.path.join(FOLDER_PATH, 'Victoria.txt')

# Function to download the file
def download_data(url=URL, folder_path=FOLDER_PATH, file_path=FILE_PATH):
    os.makedirs(folder_path, exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File successfully downloaded and saved to {file_path}")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Function to load and preprocess data
def load_and_preprocess_data(file_path=FILE_PATH):
    # Define column names
    columns = [
        'Satellite_mission_name', 'Satellite_repeat_cycle', 'Date',
        'Hour', 'Minutes', 'Target_height_variation', 'Estimated_error',
        'Mean_Ku_band_backscatter', 'Wet_tropospheric_correction',
        'Ionosphere_correction', 'Dry_tropospheric_correction',
        'Instrument_mode_1', 'Instrument_mode_2', 'Frozen_surface_flag',
        'Target_height_variation_EGM2008', 'Data_source_flag'
    ]
    
    # Load data, convert to correct formats, and clean
    victoria = pd.read_csv(file_path, skiprows=50, delim_whitespace=True, names=columns)
    victoria['Date'] = pd.to_datetime(victoria['Date'], format='%Y%m%d', errors='coerce')
    victoria['Target_height_variation'] = pd.to_numeric(victoria['Target_height_variation'], errors='coerce')
    victoria.replace([999.99, 99.999], np.nan, inplace=True)
    
    # Select columns of interest and group by date
    victoria = victoria[['Date', 'Target_height_variation']]
    victoria = victoria.groupby('Date').mean().reset_index()
    victoria.rename(columns={"Target_height_variation": "victoria_height_variation"}, inplace=True)
    
    return victoria

# Helper function to find the nearest past date in a list
def find_nearest_future_date(date, date_list):
    future_dates = [d for d in date_list if d < date]
    return future_dates[-1] if future_dates else np.nan

# Function to align with dates of interest
def align_with_dates(victoria, dates_list):
    victoria['date'] = victoria['Date'].apply(lambda x: find_nearest_future_date(x, dates_list))
    victoria = victoria.groupby('date').mean().reset_index()
    date_df = pd.DataFrame({'date': dates_list})
    victoria = pd.merge(date_df, victoria, on='date', how='left')
    victoria.set_index('date', inplace=True)
    return victoria.drop(columns=['Date'], errors='ignore')

# Function to handle missing values
def impute_missing_values(victoria):
    missing_mask = victoria.isnull().any(axis=1)
    group_id = (missing_mask != missing_mask.shift()).cumsum()
    consecutive_missing_counts = (
        victoria[missing_mask]
        .assign(group_id=group_id[missing_mask])
        .groupby('group_id')
        .size()
        .reset_index(name='count')
    )
    end_missing_streak = consecutive_missing_counts.iloc[-1]['count'] if not consecutive_missing_counts.empty else 0
    
    victoria_filled = victoria.interpolate(method='linear')
    if victoria.iloc[-1].isna()['victoria_height_variation']:
        victoria_filled.iloc[-end_missing_streak:] = np.nan
    return victoria_filled

# Function to scale the data
def scale_data(victoria):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(victoria)
    return pd.DataFrame(df_scaled, index=victoria.index, columns=victoria.columns)

# Function to apply additional imputation logic, if needed
def final_imputation(victoria, regression_length=6):
    # Placeholder for imputation logic; replace or remove if not needed
    # victoria = impute_missing_values(victoria, columns_to_impute, regression_length)
    return victoria

# Main processing function to be called in the main script
def process_victoria(dates_list):
    """Main function to download, process, and prepare Victoria lake data."""
    # Download and load data
    download_data()
    victoria = load_and_preprocess_data()
    
    # Align data with specified dates
    dates_list = pd.to_datetime(dates_list).sort_values()
    victoria = align_with_dates(victoria, dates_list)
    
    # Impute missing values
    victoria = impute_missing_values(victoria)
    
    # Filter to study period
    min_date = pd.to_datetime('2002-07-01')
    victoria = victoria[victoria.index >= min_date]
    
    # Scale the data
    victoria = scale_data(victoria)
    
    # Final imputation step
    victoria = final_imputation(victoria)
    
    return victoria
