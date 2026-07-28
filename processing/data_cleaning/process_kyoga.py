# Import system libraries
import os

# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime

# Import client libraries
import requests

# Import cleaning utils
from .. import cleaning_utils
from ..config import get_cfg

# Define constants for file URL and paths
URL = get_cfg("sources.lake_levels.kyoga_url", "https://blueice.gsfc.nasa.gov/gwm/timeseries/lake000398.10d.2.txt")
FOLDER_PATH = get_cfg("paths.downloads.lake_levels", "data/downloads/lake_levels")
FILE_PATH = os.path.join(FOLDER_PATH, 'Kyoga.txt')
KYOGA_OUTPUT_PATH = get_cfg("paths.historic.kyoga", "data/historic/kyoga.csv")

def download_data(url=URL, folder_path=FOLDER_PATH, file_path=FILE_PATH):
    """
    Download a file from a specified URL and save it to the specified folder path for Lake Kyoga.
    """
    try:
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Send request to download the file
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File successfully downloaded and saved to {file_path}")
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error in downloading data: {e}")

def load_and_preprocess_data(file_path=FILE_PATH):
    """
    Load and preprocess the Kyoga dataset from a specified file path.
    """
    try:
        # Define column names for Kyoga
        columns = [
            'Satellite_mission_name', 'Satellite_repeat_cycle', 'Date',
            'Hour', 'Minutes', 'Target_height_variation', 'Estimated_error',
            'Mean_Ku_band_backscatter', 'Wet_tropospheric_correction',
            'Ionosphere_correction', 'Dry_tropospheric_correction',
            'Instrument_mode_1', 'Instrument_mode_2', 'Frozen_surface_flag',
            'Target_height_variation_EGM2008', 'Data_source_flag'
        ]

        # Load data
        kyoga = pd.read_csv(file_path, skiprows=50, sep=r'\s+', names=columns)

        # Convert columns to correct formats
        kyoga['Date'] = pd.to_datetime(kyoga['Date'], format='%Y%m%d', errors='coerce')
        kyoga['Target_height_variation'] = pd.to_numeric(kyoga['Target_height_variation'], errors='coerce')

        # Handle missing values
        kyoga.replace([999.99, 99.999], np.nan, inplace=True)

        # Select columns of interest and group by date
        kyoga = kyoga[['Date', 'Target_height_variation']]
        kyoga = kyoga.groupby('Date').mean().reset_index()

        # Rename column
        kyoga.rename(columns={"Target_height_variation": "kyoga_height_variation"}, inplace=True)

        return kyoga
    except Exception as e:
        print(f"Error in loading and preprocessing data: {e}")
        return pd.DataFrame()

def find_nearest_future_date(date, date_list):
    """
    Find the nearest past date in a list that is less than the given date.
    """
    try:
        future_dates = [d for d in date_list if d < date]
        return future_dates[-1] if future_dates else np.nan
    except Exception as e:
        print(f"Error in finding nearest future date: {e}")
        return np.nan

def align_with_dates(kyoga, dates_list):
    """
    Align the Kyoga data with a list of dates, finding the nearest past date for each entry.
    """
    try:
        # Create new column with nearest past date
        kyoga['date'] = kyoga['Date'].apply(lambda x: find_nearest_future_date(x, dates_list))

        # Group by new date column
        kyoga = kyoga.groupby('date').mean().reset_index()

        # Merge with the original dates list to ensure all dates are covered
        date_df = pd.DataFrame({'date': dates_list})
        kyoga = pd.merge(date_df, kyoga, on='date', how='left')

        # Set date as index and drop the 'Date' column
        kyoga.set_index('date', inplace=True)
        return kyoga.drop(columns=['Date'], errors='ignore')
    except Exception as e:
        print(f"Error in aligning data with dates: {e}")
        return kyoga


def interpolate_missing(kyoga):
    """
    Function to impute past values of Kyoga data.
    """
    # Create a group identifier for consecutive missing values
    missing_mask = kyoga.isnull().any(axis=1)
    group_id = (missing_mask != missing_mask.shift()).cumsum()
    
    # Filter out the groups that are not missing and count consecutive missing values
    consecutive_missing_counts = (
        kyoga[missing_mask]
        .assign(group_id=group_id[missing_mask])
        .groupby('group_id')
        .size()
    )
    
    # Convert the counts to a DataFrame for better readability
    consecutive_missing_counts = consecutive_missing_counts.reset_index(name='count')
    end_missing_streak = consecutive_missing_counts.iloc[-1]['count']
    
    # Impute missing values based on most recent data
    kyoga_filled = kyoga.interpolate(method='linear')
    
    # Count how many consecutive rows are missing from the end
    last_streak = kyoga.isnull().iloc[::-1].all(axis=1).cumsum().max()
    
    # If there's a streak, revert those rows back to NaN
    if kyoga.iloc[-1].isna()['kyoga_height_variation']:
        kyoga_filled.iloc[-end_missing_streak:] = np.nan
    kyoga = kyoga_filled.copy()
    
    return kyoga


def update_kyoga(target_product=None):
    """
    Main function to download, process, and prepare the Kyoga lake data.
    """
    try:
        # Download and load data
        download_data()
        kyoga = load_and_preprocess_data()

        target_product = cleaning_utils.resolve_target_product(target_product)

        # Align data with target-aligned dates
        dates_list = cleaning_utils.get_dates_of_interest(target_product=target_product)
        dates_list = pd.to_datetime(dates_list).sort_values()
        kyoga = align_with_dates(kyoga, dates_list)

        # Impute missing values
        kyoga = interpolate_missing(kyoga)
        kyoga = cleaning_utils.impute_missing_values(kyoga, ['kyoga_height_variation'])

        # Filter to target study period
        min_date = pd.to_datetime(cleaning_utils.get_target_start_date(target_product=target_product))
        kyoga = kyoga[kyoga.index >= min_date]

        # Save the processed data
        kyoga.to_csv(KYOGA_OUTPUT_PATH, index=True)
        print("Kyoga data processing completed successfully.")
        
    except Exception as e:
        print(f"Error in processing Kyoga data: {e}")
