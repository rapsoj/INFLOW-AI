# Import system libraries
import os

# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime

# Import client libraries
import requests

# Import machine learning libraries
from sklearn.preprocessing import StandardScaler

# Import cleaning utils
from .. import cleaning_utils

# Define constants for file URL and paths
URL = 'https://blueice.gsfc.nasa.gov/gwm/timeseries/lake000314.10d.2.txt'
FOLDER_PATH = 'data/downloads/lake_levels'
FILE_PATH = os.path.join(FOLDER_PATH, 'Victoria.txt')

def download_data(url=URL, folder_path=FOLDER_PATH, file_path=FILE_PATH):
    """
    Download a file from a specified URL and save it to the specified folder path.

    Parameters:
        url (str): The URL of the file to be downloaded.
        folder_path (str): The path to the folder where the file should be saved.
        file_path (str): The file path for saving the downloaded file.

    Returns:
        None
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
    Load and preprocess the Victoria dataset from a specified file path.

    Parameters:
        file_path (str): The path to the Victoria data file.

    Returns:
        pd.DataFrame: The preprocessed Victoria data.
    """
    try:
        # Define column names
        columns = [
            'Satellite_mission_name', 'Satellite_repeat_cycle', 'Date',
            'Hour', 'Minutes', 'Target_height_variation', 'Estimated_error',
            'Mean_Ku_band_backscatter', 'Wet_tropospheric_correction',
            'Ionosphere_correction', 'Dry_tropospheric_correction',
            'Instrument_mode_1', 'Instrument_mode_2', 'Frozen_surface_flag',
            'Target_height_variation_EGM2008', 'Data_source_flag'
        ]

        # Load data
        victoria = pd.read_csv(file_path, skiprows=50, sep=r'\s+', names=columns)

        # Convert columns to correct formats
        victoria['Date'] = pd.to_datetime(victoria['Date'], format='%Y%m%d', errors='coerce')
        victoria['Target_height_variation'] = pd.to_numeric(victoria['Target_height_variation'], errors='coerce')

        # Handle missing values
        victoria.replace([999.99, 99.999], np.nan, inplace=True)

        # Select columns of interest and group by date
        victoria = victoria[['Date', 'Target_height_variation']]
        victoria = victoria.groupby('Date').mean().reset_index()

        # Rename column
        victoria.rename(columns={"Target_height_variation": "victoria_height_variation"}, inplace=True)

        return victoria
    except Exception as e:
        print(f"Error in loading and preprocessing data: {e}")
        return pd.DataFrame()


def find_nearest_future_date(date, date_list):
    """
    Find the nearest past date in a list that is less than the given date.

    Parameters:
        date (datetime): The target date.
        date_list (list): A list of dates to compare against.

    Returns:
        datetime: The nearest past date.
    """
    try:
        future_dates = [d for d in date_list if d < date]
        return future_dates[-1] if future_dates else np.nan
    except Exception as e:
        print(f"Error in finding nearest future date: {e}")
        return np.nan


def align_with_dates(victoria, dates_list):
    """
    Align the Victoria data with a list of dates, finding the nearest past date for each entry.

    Parameters:
        victoria (pd.DataFrame): The Victoria dataset.
        dates_list (list): A list of dates to align with.

    Returns:
        pd.DataFrame: The Victoria dataset aligned with the provided dates.
    """
    try:
        # Create new column with nearest past date
        victoria['date'] = victoria['Date'].apply(lambda x: find_nearest_future_date(x, dates_list))

        # Group by new date column
        victoria = victoria.groupby('date').mean().reset_index()

        # Merge with the original dates list to ensure all dates are covered
        date_df = pd.DataFrame({'date': dates_list})
        victoria = pd.merge(date_df, victoria, on='date', how='left')

        # Set date as index and drop the 'Date' column
        victoria.set_index('date', inplace=True)
        return victoria.drop(columns=['Date'], errors='ignore')
    except Exception as e:
        print(f"Error in aligning data with dates: {e}")
        return victoria


def scale_data(victoria):
    """
    Scale the Victoria dataset using StandardScaler based on the first 804 rows.

    Parameters:
        victoria (pd.DataFrame): The Victoria dataset.

    Returns:
        pd.DataFrame: The scaled Victoria dataset.
    """
    try:
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler on the first 804 rows
        scaler.fit(victoria.iloc[:804])

        # Transform the entire dataset using the fitted scaler
        df_scaled = scaler.transform(victoria)

        return pd.DataFrame(df_scaled, index=victoria.index, columns=victoria.columns)
    except Exception as e:
        print(f"Error in scaling data: {e}")
        return victoria
        
        
def interpolate_missing(victoria):
    """
    Function to impute past values of Victoria data.
    """
    # Create a group identifier for consecutive missing values
    missing_mask = victoria.isnull().any(axis=1)
    group_id = (missing_mask != missing_mask.shift()).cumsum()
    
    # Filter out the groups that are not missing and count consecutive missing values
    consecutive_missing_counts = (
        victoria[missing_mask]
        .assign(group_id=group_id[missing_mask])
        .groupby('group_id')
        .size()
    )
    
    # Convert the counts to a DataFrame for better readability
    consecutive_missing_counts = consecutive_missing_counts.reset_index(name='count')
    end_missing_streak = consecutive_missing_counts.iloc[-1]['count']
    
    # Impute missing values based on most recent data
    victoria_filled = victoria.interpolate(method='linear')
    
    # Count how many consecutive rows are missing from the end
    last_streak = victoria.isnull().iloc[::-1].all(axis=1).cumsum().max()
    
    # If there's a streak, revert those rows back to NaN
    if victoria.iloc[-1].isna()['victoria_height_variation']:
        victoria_filled.iloc[-end_missing_streak:] = np.nan
    victoria = victoria_filled.copy()
    
    return victoria


def update_victoria():
    """
    Main function to download, process, and prepare the Victoria lake data.

    Parameters:
        dates_list (list): A list of dates to align with.

    Returns:
        pd.DataFrame: The processed and prepared Victoria dataset.
    """
    try:
        # Download and load data
        download_data()
        victoria = load_and_preprocess_data()

        # Align data with the specified dates
        dates_list = cleaning_utils.get_dates_of_interest()
        dates_list = pd.to_datetime(dates_list).sort_values()
        victoria = align_with_dates(victoria, dates_list)

        # Impute missing values
        victoria = interpolate_missing(victoria)
        victoria = cleaning_utils.impute_missing_values(victoria, ['victoria_height_variation'])

        # Filter to study period
        min_date = pd.to_datetime('2002-07-01')
        victoria = victoria[victoria.index >= min_date]

        # Scale the data
        victoria = scale_data(victoria)

        # Save the processed data
        victoria.to_csv('data/historic/victoria.csv', index=True)
        print("Victoria data processing completed successfully.")

    except Exception as e:
        print(f"Error in processing Victoria data: {e}")