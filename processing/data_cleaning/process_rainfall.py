# Import system libraries
import os

# Import data manipulation libraries
import pandas as pd
import numpy as np

# Import client libraries
import requests

# Import machine learning libraries
from sklearn.preprocessing import StandardScaler

# Import cleaning utils
from .. import cleaning_utils

# Define constants
URL = "https://gws-access.jasmin.ac.uk/public/tamsat/INFLOW/rainfall/rfe_time-series/combined/rfe_19830101-present_Lake-Victoria.csv"
FOLDER_PATH = "data/downloads/tamsat"
FILE_NAME = "rainfall.csv"
FILE_PATH = os.path.join(FOLDER_PATH, FILE_NAME)


def download_rainfall_data(url=URL, folder_path=FOLDER_PATH, file_path=FILE_PATH):
    """
    Download the rainfall data CSV from the provided URL and save it to the specified folder path.
    
    Parameters:
        url (str): The URL of the file to be downloaded.
        folder_path (str): The path to the folder where the file should be saved.
        file_path (str): The full path for saving the downloaded file.
        
    Returns:
        None
    """
    try:
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Send a request to download the file
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            raise Exception(f"Failed to download the file. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error in downloading data: {e}")


def load_and_preprocess_rainfall_data(file_path=FILE_PATH):
    """
    Load the rainfall data from the CSV file, and preprocess it (convert dates and handle missing values).
    
    Parameters:
        file_path (str): The path to the rainfall data CSV file.
        
    Returns:
        pd.DataFrame: The preprocessed rainfall data.
    """
    try:
        # Load the data
        rainfall = pd.read_csv(file_path)

        # Convert time column to datetime format
        rainfall['time'] = pd.to_datetime(rainfall['time'])

        # Return the preprocessed data
        return rainfall
    except Exception as e:
        print(f"Error in loading and preprocessing data: {e}")
        return pd.DataFrame()


def find_nearest_future_date(date, date_list):
    """
    Find the nearest future date from a list of dates that is less than the given date.
    
    Parameters:
        date (datetime): The target date to compare.
        date_list (list): A list of dates to compare against.
        
    Returns:
        datetime: The nearest future date.
    """
    future_dates = [d for d in date_list if d < date]
    return future_dates[-1] if future_dates else np.nan


def align_rainfall_with_dates(rainfall, dates_list):
    """
    Align the rainfall data with a list of dates, finding the nearest past date for each entry.
    
    Parameters:
        rainfall (pd.DataFrame): The rainfall dataset.
        dates_list (list): A list of dates to align with.
        
    Returns:
        pd.DataFrame: The rainfall dataset aligned with the provided dates.
    """
    try:
        # Get sorted list of dates
        dates_list = pd.to_datetime(dates_list).sort_values()

        # Create new column with nearest past date
        rainfall['date'] = rainfall['time'].apply(lambda x: find_nearest_future_date(x, dates_list))

        # Group by new date column and calculate mean
        rainfall = rainfall.groupby('date').mean().reset_index()

        # Merge with the original dates list to ensure all dates are covered
        date_df = pd.DataFrame({'date': dates_list})
        rainfall = pd.merge(date_df, rainfall, on='date', how='left')

        # Set date as index
        rainfall = rainfall.sort_values('date').set_index('date')

        # Drop the original 'time' column
        rainfall = rainfall.drop('time', axis=1)
        
        # Select columns of interest
        rainfall = rainfall[['TAMSAT', 'CHIRPS']]

        return rainfall
    except Exception as e:
        print(f"Error in aligning data with dates: {e}")
        return rainfall


def calculate_cumulative_values(rainfall, columns):
    """
    Calculate cumulative values from scaled original values for the specified columns.
    
    Parameters:
        rainfall (pd.DataFrame): The rainfall dataset.
        columns (list): List of column names to calculate cumulative values for.
        
    Returns:
        pd.DataFrame: The rainfall dataset with cumulative columns added.
    """
    try:
        for col in columns:
            rainfall[col + '_cumulative'] = ((rainfall[col] - rainfall[col].mean()) / rainfall[col].std()).cumsum()
        
        return rainfall
    except Exception as e:
        print(f"Error in calculating cumulative values: {e}")
        return rainfall


def scale_data(rainfall):
    """
    Scale the rainfall dataset using StandardScaler based on the first 804 rows.

    Parameters:
        rainfall (pd.DataFrame): The rainfall dataset.

    Returns:
        pd.DataFrame: The scaled rainfall dataset.
    """
    try:
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler on the first 804 rows
        scaler.fit(rainfall.iloc[:804])

        # Transform the entire dataset using the fitted scaler
        df_scaled = scaler.transform(rainfall)

        return pd.DataFrame(df_scaled, index=rainfall.index, columns=rainfall.columns)
    except Exception as e:
        print(f"Error in scaling data: {e}")
        return rainfall
        
        
def interpolate_missing(rainfall):
    """
    Function to impute past values of rainfall data.
    """
    # Iterate through each column in the dataframe
    for col in rainfall.columns:
    
        # Create a mask for missing values in the current column
        missing_mask = rainfall[col].isnull()
        group_id = (missing_mask != missing_mask.shift()).cumsum()
    
        # Filter out the groups that are not missing and count consecutive missing values
        consecutive_missing_counts = (
            rainfall[missing_mask]
            .assign(group_id=group_id[missing_mask])
            .groupby('group_id')
            .size()
        )
    
        # Convert the counts to a DataFrame for better readability
        consecutive_missing_counts = consecutive_missing_counts.reset_index(name='count')
        end_missing_streak = consecutive_missing_counts.iloc[-1]['count'] if not consecutive_missing_counts.empty else 0
    
        # Impute missing values for the current column using linear interpolation
        rainfall_filled = rainfall.copy()
        rainfall_filled[col] = rainfall[col].interpolate(method='linear')
    
        # Count how many consecutive rows are missing from the end in the current column
        last_streak = rainfall[col].isnull().iloc[::-1].cumsum().max()
    
        # If there's a streak of missing values at the end, revert those rows back to NaN
        if last_streak > 0:
            last_indices = rainfall_filled.index[-end_missing_streak:]
            rainfall_filled.loc[last_indices, col] = np.nan
    
        # Update the original DataFrame with the filled values
        rainfall[col] = rainfall_filled[col]
        
    return rainfall


def update_rainfall():
    """
    Main function to download, process, and prepare the rainfall data for deployment.

    Parameters:
        dates_list (list): A list of dates to align with.
        
    Returns:
        pd.DataFrame: The processed and prepared rainfall dataset.
    """
    try:
        # Download and load data
        download_rainfall_data()
        rainfall = load_and_preprocess_rainfall_data()

        # Align data with the specified dates
        dates_list = cleaning_utils.get_dates_of_interest()
        dates_list = pd.to_datetime(dates_list).sort_values()
        rainfall = align_rainfall_with_dates(rainfall, dates_list)

        # Impute missing values
        rainfall = interpolate_missing(rainfall)
        rainfall = cleaning_utils.impute_missing_values(rainfall, ['TAMSAT', 'CHIRPS'])

        # Calculate cumulative values
        rainfall = calculate_cumulative_values(rainfall, ['TAMSAT', 'CHIRPS'])

        # Filter to study period
        min_date = pd.to_datetime('2002-07-01')
        rainfall = rainfall[rainfall.index >= min_date]

        # Scale the data
        rainfall = scale_data(rainfall)

        # Save the processed data
        rainfall.to_csv('data/historic/rainfall.csv', index=True)
        print("Rainfall data processing completed successfully.")

    except Exception as e:
        print(f"Error in processing rainfall data: {e}")
