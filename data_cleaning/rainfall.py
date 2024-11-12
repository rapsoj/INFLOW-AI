import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from processing import get_dates_interest, impute_missing_values  # Import necessary functions from processing.py

# Constants for file paths
FOLDER_PATH = 'Rainfall'
FILE_NAME = 'rfe_19830101-present_Lake-Victoria.csv'
FILE_PATH = os.path.join(FOLDER_PATH, FILE_NAME)

# Function to download the rainfall data
def download_rainfall_data(url):
    """Download the rainfall data from the specified URL."""
    os.makedirs(FOLDER_PATH, exist_ok=True)
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(FILE_PATH, 'wb') as file:
            file.write(response.content)
        print(f"File successfully downloaded and saved to {FILE_PATH}")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Function to load and preprocess rainfall data
def load_and_preprocess_rainfall(file_path=FILE_PATH):
    """Load and preprocess the rainfall data."""
    # Load data from the CSV file
    rainfall = pd.read_csv(file_path)

    # Convert the 'time' column to datetime format
    rainfall['time'] = pd.to_datetime(rainfall['time'])
    
    return rainfall

# Helper function to find the nearest future date in a list of dates
def find_nearest_future_date(date, date_list):
    """Find the nearest past date from the list of dates."""
    future_dates = [d for d in date_list if d < date]
    return future_dates[-1] if future_dates else np.nan

# Function to align rainfall data with a list of dates
def align_rainfall_with_dates(rainfall, dates_list):
    """Align rainfall data with the given list of dates."""
    # Convert the list of dates to a sorted pandas datetime object
    dates_list = pd.to_datetime(dates_list).sort_values()
    
    # Create new column with the nearest future date
    rainfall['date'] = rainfall['time'].apply(lambda x: find_nearest_future_date(x, dates_list))

    # Filter the data for the study period
    min_date = rainfall['date'].min()
    previous_min_date = rainfall[rainfall['time'] <= min_date].sort_values('time').reset_index().iloc[-1]['time']
    rainfall = rainfall[rainfall['time'] >= previous_min_date]

    # Group by 'date' and calculate the mean for each group
    rainfall = rainfall.groupby('date').mean().reset_index()

    # Merge the rainfall data with the list of ordered dates
    date_df = pd.DataFrame({'date': dates_list})
    rainfall = pd.merge(date_df, rainfall, on='date', how='left')

    # Set 'date' as the index and remove 'time' column
    rainfall.set_index('date', inplace=True)
    return rainfall.drop(columns=['time'], errors='ignore')

# Function to handle missing values using linear interpolation
def impute_missing_values_rainfall(rainfall):
    """Impute missing values in rainfall data using linear interpolation."""
    for col in rainfall.columns:
        missing_mask = rainfall[col].isnull()
        group_id = (missing_mask != missing_mask.shift()).cumsum()

        # Count consecutive missing values
        consecutive_missing_counts = (
            rainfall[missing_mask]
            .assign(group_id=group_id[missing_mask])
            .groupby('group_id')
            .size()
        )
        consecutive_missing_counts = consecutive_missing_counts.reset_index(name='count')
        end_missing_streak = consecutive_missing_counts.iloc[-1]['count'] if not consecutive_missing_counts.empty else 0

        # Impute missing values using linear interpolation
        rainfall_filled = rainfall.copy()
        rainfall_filled[col] = rainfall[col].interpolate(method='linear')

        # Revert missing values at the end of the data to NaN
        last_streak = rainfall[col].isnull().iloc[::-1].cumsum().max()
        if last_streak > 0:
            last_indices = rainfall_filled.index[-end_missing_streak:]
            rainfall_filled.loc[last_indices, col] = np.nan

        # Update the original DataFrame with the filled values
        rainfall[col] = rainfall_filled[col]
    
    return rainfall

# Function to calculate cumulative values for rainfall data
def calculate_cumulative_values(rainfall):
    """Calculate cumulative values for the 'TAMSAT' and 'CHIRPS' columns."""
    cumulative_cols = ['TAMSAT', 'CHIRPS']
    for col in cumulative_cols:
        rainfall[col + '_cumulative'] = ((rainfall[col] - rainfall[col].mean()) / rainfall[col].std()).cumsum()

    return rainfall

# Function to scale the rainfall data
def scale_rainfall_data(rainfall):
    """Scale the rainfall data using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(rainfall)
    return pd.DataFrame(df_scaled, index=rainfall.index, columns=rainfall.columns)

# Main function to process the rainfall data
def process_rainfall(dates_list):
    """Main function to download, process, and prepare rainfall data."""
    # URL to download the rainfall data
    url = "https://gws-access.jasmin.ac.uk/public/tamsat/INFLOW/rainfall/rfe_time-series/combined/rfe_19830101-present_Lake-Victoria.csv"
    
    # Download the rainfall data
    download_rainfall_data(url)

    # Load and preprocess rainfall data
    rainfall = load_and_preprocess_rainfall()

    # Align rainfall data with the list of interest dates
    rainfall = align_rainfall_with_dates(rainfall, dates_list)

    # Impute missing values
    rainfall = impute_missing_values_rainfall(rainfall)

    # Calculate cumulative values
    rainfall = calculate_cumulative_values(rainfall)

    # Filter data to study period starting from July 1st, 2002
    min_date = pd.to_datetime('2002-07-01')
    rainfall = rainfall[rainfall.index >= min_date]

    # Scale the data
    rainfall = scale_rainfall_data(rainfall)

    # Final imputation of missing values
    columns_to_impute = rainfall.columns.tolist()
    rainfall = impute_missing_values(rainfall, columns_to_impute, regression_length=6)

    return rainfall
