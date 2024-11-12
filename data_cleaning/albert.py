import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from processing import get_dates_interest, impute_missing_values  # Import necessary functions from processing.py

# Constants for file paths
FOLDER_PATH = 'Satellite Altimetry etc'
FILE_PATH = os.path.join(FOLDER_PATH, 'Albert.txt')

# Function to load and preprocess Albert data
def load_and_preprocess_data(file_path=FILE_PATH):
    # Define column names
    columns = [
        "decimal_year", "measurement_date", "time", "height", "std", "area", "volume", "mission"]

    # Load the dataset
    albert = pd.read_csv(file_path, skiprows=47, sep=';', names=columns, index_col=False)

    # Convert 'measurement_date' to datetime
    albert['measurement_date'] = pd.to_datetime(albert['measurement_date']).dt.strftime('%Y-%m-%d')
    albert['measurement_date'] = pd.to_datetime(albert['measurement_date'])

    # Filter relevant columns and calculate average height per day
    albert = albert[['measurement_date', 'height']]
    albert = albert.groupby('measurement_date').mean().reset_index()

    # Rename the 'height' column
    albert = albert.rename(columns={"height": "albert_water_level"})
    
    return albert

# Helper function to find the nearest past date in the list of dates
def find_nearest_future_date(date, date_list):
    future_dates = [d for d in date_list if d < date]
    return future_dates[-1] if future_dates else np.nan

# Function to align the data with the list of interest dates
def align_with_dates(albert, dates_list):
    albert['date'] = albert['measurement_date'].apply(lambda x: find_nearest_future_date(x, dates_list))
    albert = albert.groupby('date').mean().reset_index()

    # Create a DataFrame for the dates and merge with the data
    date_df = pd.DataFrame({'date': dates_list})
    albert = pd.merge(date_df, albert, on='date', how='left')

    # Set the 'date' as the index
    albert.set_index('date', inplace=True)
    return albert.drop(columns=['measurement_date'], errors='ignore')

# Function to handle missing values using linear interpolation
def impute_missing_values_v2(albert):
    missing_mask = albert.isnull().any(axis=1)
    group_id = (missing_mask != missing_mask.shift()).cumsum()

    # Count consecutive missing values
    consecutive_missing_counts = (
        albert[missing_mask]
        .assign(group_id=group_id[missing_mask])
        .groupby('group_id')
        .size()
        .reset_index(name='count')
    )
    end_missing_streak = consecutive_missing_counts.iloc[-1]['count'] if not consecutive_missing_counts.empty else 0

    # Impute missing values using linear interpolation
    albert_filled = albert.interpolate(method='linear')
    if albert.iloc[-1].isna()['albert_water_level']:
        albert_filled.iloc[-end_missing_streak:] = np.nan
    return albert_filled

# Function to scale the data using StandardScaler
def scale_data(albert):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(albert)
    return pd.DataFrame(df_scaled, index=albert.index, columns=albert.columns)

# Main processing function to be called in the main script
def process_albert(dates_list):
    """Main function to load, process, and prepare Albert lake data."""
    # Load and preprocess data
    albert = load_and_preprocess_data()

    # Align data with the list of interest dates
    dates_list = pd.to_datetime(dates_list).sort_values()
    albert = align_with_dates(albert, dates_list)

    # Impute missing values
    albert = impute_missing_values_v2(albert)

    # Filter to study period (starting from July 1st, 2002)
    min_date = pd.to_datetime('2002-07-01')
    albert = albert[albert.index >= min_date]

    # Scale the data
    albert = scale_data(albert)

    # Final imputation step (if needed)
    columns_to_impute = albert.columns.tolist()
    albert = impute_missing_values(albert, columns_to_impute, regression_length=6)

    return albert