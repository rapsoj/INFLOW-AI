import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from processing import get_dates_interest, impute_missing_values  # Import necessary functions from processing.py

# Constants for file paths
FOLDER_PATH = 'Satellite Altimetry etc'
FILE_PATH = os.path.join(FOLDER_PATH, 'Kyoga.txt')

# Function to download the Kyoga data file
def download_data(url):
    """Download the Kyoga data from the specified URL."""
    os.makedirs(FOLDER_PATH, exist_ok=True)
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(FILE_PATH, 'wb') as file:
            file.write(response.content)
        print(f"File successfully downloaded and saved to {FILE_PATH}")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Function to load and preprocess Kyoga data
def load_and_preprocess_data(file_path=FILE_PATH):
    """Load and preprocess the Kyoga data."""
    columns = [
        'Satellite_mission_name', 'Satellite_repeat_cycle', 'Date',
        'Hour', 'Minutes', 'Target_height_variation', 'Estimated_error',
        'Mean_Ku_band_backscatter', 'Wet_tropospheric_correction',
        'Ionosphere_correction', 'Dry_tropospheric_correction',
        'Instrument_mode_1', 'Instrument_mode_2', 'Frozen_surface_flag',
        'Target_height_variation_EGM2008', 'Data_source_flag'
    ]
    
    # Load the dataset
    kyoga = pd.read_csv(file_path, skiprows=50, delim_whitespace=True, names=columns)

    # Convert 'Date' to datetime and 'Target_height_variation' to numeric
    kyoga['Date'] = pd.to_datetime(kyoga['Date'], format='%Y%m%d', errors='coerce')
    kyoga['Target_height_variation'] = pd.to_numeric(kyoga['Target_height_variation'], errors='coerce')

    # Replace missing data codes (999.99 and 99.999) with NaN
    kyoga = kyoga.replace(999.99, np.nan)
    kyoga = kyoga.replace(99.999, np.nan)

    # Select columns of interest and group by date
    kyoga = kyoga[['Date', 'Target_height_variation']]
    kyoga = kyoga.groupby('Date').mean().reset_index()

    # Rename 'Target_height_variation' column
    kyoga = kyoga.rename(columns={"Target_height_variation": "kyoga_height_variation"})
    
    return kyoga

# Helper function to find the nearest future date in a list of dates
def find_nearest_future_date(date, date_list):
    """Find the nearest past date from the list of dates."""
    future_dates = [d for d in date_list if d < date]
    return future_dates[-1] if future_dates else np.nan

# Function to align Kyoga data with a given list of dates
def align_with_dates(kyoga, dates_list):
    """Align Kyoga data with the given list of dates."""
    kyoga['date'] = kyoga['Date'].apply(lambda x: find_nearest_future_date(x, dates_list))
    kyoga = kyoga.groupby('date').mean().reset_index()

    # Merge with the ordered list of dates
    date_df = pd.DataFrame({'date': dates_list})
    kyoga = pd.merge(date_df, kyoga, on='date', how='left')

    # Set 'date' as the index and drop the 'Date' column
    kyoga.set_index('date', inplace=True)
    return kyoga.drop(columns=['Date'], errors='ignore')

# Function to handle missing values using linear interpolation
def impute_missing_values_v2(kyoga):
    """Impute missing values in Kyoga data using linear interpolation."""
    missing_mask = kyoga.isnull().any(axis=1)
    group_id = (missing_mask != missing_mask.shift()).cumsum()

    # Count consecutive missing values
    consecutive_missing_counts = (
        kyoga[missing_mask]
        .assign(group_id=group_id[missing_mask])
        .groupby('group_id')
        .size()
        .reset_index(name='count')
    )
    end_missing_streak = consecutive_missing_counts.iloc[-1]['count'] if not consecutive_missing_counts.empty else 0

    # Impute missing values using linear interpolation
    kyoga_filled = kyoga.interpolate(method='linear')
    if kyoga.iloc[-1].isna()['kyoga_height_variation']:
        kyoga_filled.iloc[-end_missing_streak:] = np.nan
    return kyoga_filled

# Function to scale the Kyoga data using StandardScaler
def scale_data(kyoga):
    """Scale the Kyoga data using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(kyoga)
    return pd.DataFrame(df_scaled, index=kyoga.index, columns=kyoga.columns)

# Main processing function to be called in the main script
def process_kyoga(dates_list):
    """Main function to load, process, and prepare Kyoga lake data."""
    # Download the Kyoga data (if needed)
    url = 'https://blueice.gsfc.nasa.gov/gwm/timeseries/lake000398.10d.2.txt'
    download_data(url)

    # Load and preprocess Kyoga data
    kyoga = load_and_preprocess_data()

    # Align data with the list of interest dates
    dates_list = pd.to_datetime(dates_list).sort_values()
    kyoga = align_with_dates(kyoga, dates_list)

    # Impute missing values
    kyoga = impute_missing_values_v2(kyoga)

    # Filter to study period (starting from July 1st, 2002)
    min_date = pd.to_datetime('2002-07-01')
    kyoga = kyoga[kyoga.index >= min_date]

    # Scale the data
    kyoga = scale_data(kyoga)

    # Final imputation step (if needed)
    columns_to_impute = kyoga.columns.tolist()
    kyoga = impute_missing_values(kyoga, columns_to_impute, regression_length=6)

    return kyoga
