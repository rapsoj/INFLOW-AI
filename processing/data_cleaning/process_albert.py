# Import system libraries
import os
import sys
import zipfile

# Import data manipulation libraries
import pandas as pd
import numpy as np

# Import client libraries
import py_hydroweb

# Import logging libraries
import logging

# Import machine learning libraries
from sklearn.preprocessing import StandardScaler

# Import cleaning utils
from .. import cleaning_utils

# Set log config
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Create a client
client: py_hydroweb.Client = py_hydroweb.Client("https://hydroweb.next.theia-land.fr/api", api_key="70s88Dpwc0UraJi9n4EouE7UXCbXzQJkSdCeT1cLNy4EcczioA")

# Define function to download and extract data
def download_and_extract_data(bounding_box, output_file_path, zip_filename="data/downloads/lake_levels/albert.zip"):
    """
    Download the dataset for Lake Albert using py_hydroweb API and extract the relevant file to the desired output path.
    """
    basket = py_hydroweb.DownloadBasket("my_download_basket")
    basket.add_collection("HYDROWEB_LAKES_OPE", bbox=bounding_box)
    
    downloaded_zip_path = client.submit_and_download_zip(
        basket, zip_filename=zip_filename)
    
    with zipfile.ZipFile(downloaded_zip_path, 'r') as zip_ref:
        files_in_zip = zip_ref.namelist()
        print(f"Files in zip: {files_in_zip}")

        target_file = "HYDROWEB_LAKES_OPE/HYDROWEB_LAKES_OPE/hydroprd_L_albert.txt"
        if target_file in files_in_zip:
            extracted_path = zip_ref.extract(target_file, path="data/downloads/lake_levels/temp_extraction")
            final_path = os.path.join(os.getcwd(), output_file_path)
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            os.rename(extracted_path, final_path)
            print(f"File extracted and renamed to: {final_path}")
        else:
            print(f"Target file {target_file} not found in zip.")
    
    if os.path.exists(downloaded_zip_path):
        os.remove(downloaded_zip_path)
        print(f"Zip file {downloaded_zip_path} deleted.")
    else:
        print(f"Zip file {downloaded_zip_path} not found for deletion.")

    return final_path


# Define function to load and preprocess data
def load_and_preprocess_data(file_path, skiprows=47):
    """
    Load and preprocess the Lake Albert dataset, including handling dates and renaming columns.
    """
    columns = ["decimal_year", "measurement_date", "time", "height", "std", "area", "volume", "mission"]
    albert = pd.read_csv(file_path, skiprows=skiprows, sep=';', names=columns, index_col=False)

    # Convert date and rename columns
    albert['measurement_date'] = pd.to_datetime(albert['measurement_date']).dt.strftime('%Y-%m-%d')
    albert['measurement_date'] = pd.to_datetime(albert['measurement_date'])
    albert = albert[['measurement_date', 'height']]
    albert = albert.groupby('measurement_date').mean().reset_index()
    albert = albert.rename(columns={"height": "albert_water_level"})
    
    return albert


# Define function to align with specific dates
def align_with_dates(albert, dates_list):
    """
    Align the dataset with a list of dates, filling missing dates with NaN.
    """
    def find_nearest_future_date(date, date_list):
        future_dates = [d for d in date_list if d < date]
        if future_dates:
            return future_dates[-1]
        else:
            return np.nan

    dates_list = pd.to_datetime(dates_list).sort_values()
    albert['date'] = albert['measurement_date'].apply(lambda x: find_nearest_future_date(x, dates_list))
    albert = albert.groupby('date').mean().reset_index()
    
    date_df = pd.DataFrame({'date': dates_list})
    date_df['date'] = pd.to_datetime(date_df['date'])
    albert = pd.merge(date_df, albert, on='date', how='left')
    
    albert = albert.sort_values('date')
    albert = albert.set_index('date')
    albert = albert.drop('measurement_date', axis=1)

    return albert


def scale_data(albert):
    """
    Scale the Albert dataset using StandardScaler based on the first 804 rows.

    Parameters:
        albert (pd.DataFrame): The Albert dataset.

    Returns:
        pd.DataFrame: The scaled Albert dataset.
    """
    try:
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler on the first 804 rows
        scaler.fit(albert.iloc[:804])

        # Transform the entire dataset using the fitted scaler
        df_scaled = scaler.transform(albert)

        return pd.DataFrame(df_scaled, index=albert.index, columns=albert.columns)
    except Exception as e:
        print(f"Error in scaling data: {e}")
        return albert
        
        
def interpolate_missing(albert):
    """
    Function to impute past values of Albert data.
    """
    # Create a group identifier for consecutive missing values
    missing_mask = albert.isnull().any(axis=1)
    group_id = (missing_mask != missing_mask.shift()).cumsum()
    
    # Filter out the groups that are not missing and count consecutive missing values
    consecutive_missing_counts = (
        albert[missing_mask]
        .assign(group_id=group_id[missing_mask])
        .groupby('group_id')
        .size()
    )
    
    # Convert the counts to a DataFrame for better readability
    consecutive_missing_counts = consecutive_missing_counts.reset_index(name='count')
    end_missing_streak = consecutive_missing_counts.iloc[-1]['count']
    
    # Impute missing values based on most recent data
    albert_filled = albert.interpolate(method='linear')
    
    # Count how many consecutive rows are missing from the end
    last_streak = albert.isnull().iloc[::-1].all(axis=1).cumsum().max()
    
    # If there's a streak, revert those rows back to NaN
    if albert.iloc[-1].isna()['albert_water_level']:
        albert_filled.iloc[-end_missing_streak:] = np.nan
    albert = albert_filled.copy()

    return albert

# Define main function to process Lake Albert data
def update_albert():
    """
    Main function to process Lake Albert data from download to final output.
    Handles errors gracefully so failures won't crash the entire program.
    """
    try:
        # Define bounding box for Lake Albert
        bounding_box = [30.5, 1.5, 31.5, 2.5]
        output_file_path = "data/downloads/lake_levels/Albert.txt"

        # Download and extract data
        file_path = download_and_extract_data(bounding_box, output_file_path)

        # Load and preprocess data
        albert = load_and_preprocess_data(file_path)

        # Align data with the specified dates
        dates_list = cleaning_utils.get_dates_of_interest()
        dates_list = pd.to_datetime(dates_list).sort_values()
        albert = align_with_dates(albert, dates_list)

        # Impute missing values
        albert = interpolate_missing(albert)
        albert = cleaning_utils.impute_missing_values(albert, ['albert_water_level'])
        
        # Filter to study period
        min_date = pd.to_datetime('2002-07-01')
        albert = albert[albert.index >= min_date]
        
        # Scale the data
        albert = scale_data(albert)

        # Save processed data
        albert.to_csv('data/historic/albert.csv', index=True)
        print("Lake Albert data processing completed successfully.")
    
    except Exception as e:
        # Log the error but don’t raise it
        print(f"⚠️ Lake Albert data update failed: {e}")
