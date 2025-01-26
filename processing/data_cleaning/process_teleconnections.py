# Import data manipulation libraries
import pandas as pd
import numpy as np
from functools import reduce

# Import machine learning libraries
from sklearn.preprocessing import StandardScaler

# Import logging libraries
import loguru

# Import cleaning utils
from .. import cleaning_utils

# Import teleconnections download files
from processing.data_cleaning.download_teleconnections import oni
from processing.data_cleaning.download_teleconnections import sst
from processing.data_cleaning.download_teleconnections import soi
from processing.data_cleaning.download_teleconnections import mjo
from processing.data_cleaning.download_teleconnections import dmi


def add_date_columns(teleconnections_dfs):
    """
    Add 'date_reported' and 'date_represented' columns to each teleconnections dataset.
    """
    for i in range(len(teleconnections_dfs)):
        df = teleconnections_dfs[i]
        if 'day' not in df.columns:
            df['day'] = 25
            df['date_represented'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str))
            df = df.sort_values('date_represented')
            df['date_reported'] = df['date_represented'] + pd.DateOffset(months=1)
        else:
            df['date_represented'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str))
            df = df.sort_values('date_represented')
            df['date_reported'] = df['date_represented'] + pd.DateOffset(days=5)
        teleconnections_dfs[i] = df.drop(['year', 'month', 'day'], axis=1)

    return teleconnections_dfs


def merge_teleconnections_data(teleconnections_dfs):
    """
    Merge all the teleconnections data into one dataframe.
    """
    teleconnections = reduce(lambda left, right: pd.merge(left, right, on=['date_represented', 'date_reported'], how='outer'), teleconnections_dfs)
    return teleconnections


def find_nearest_future_date(date, date_list):
    """
    Find the nearest future date from a list of dates.
    """
    future_dates = [d for d in date_list if d < date]
    if future_dates:
        return future_dates[-1]
    else:
        return np.nan


def align_teleconnections_with_dates(teleconnections, dates_list):
    """
    Align the teleconnections data with a list of dates.
    """
    dates_list = pd.to_datetime(dates_list).sort_values()
    teleconnections['date'] = teleconnections['date_represented'].apply(lambda x: find_nearest_future_date(x, dates_list))

    # Calculate average data for each date
    teleconnections = teleconnections.groupby('date').mean().reset_index()

    # Match to ordered dates
    date_df = pd.DataFrame({'date': dates_list})
    date_df['date'] = pd.to_datetime(date_df['date'])
    teleconnections = pd.merge(date_df, teleconnections, on='date', how='left')

    # Set date column as index
    teleconnections = teleconnections.sort_values('date')
    teleconnections = teleconnections.set_index('date')

    # Remove extra columns
    teleconnections = teleconnections.drop(['date_represented', 'date_reported'], axis=1)

    return teleconnections


def handle_missing_values(teleconnections):
    """
    Handle missing values in teleconnections data by applying linear interpolation.
    """
    for col in teleconnections.columns:
        # Create a mask for missing values in the current column
        missing_mask = teleconnections[col].isnull()
        group_id = (missing_mask != missing_mask.shift()).cumsum()

        # Filter out the groups that are not missing and count consecutive missing values
        consecutive_missing_counts = (
            teleconnections[missing_mask]
            .assign(group_id=group_id[missing_mask])
            .groupby('group_id')
            .size()
        )

        consecutive_missing_counts = consecutive_missing_counts.reset_index(name='count')
        end_missing_streak = consecutive_missing_counts.iloc[-1]['count'] if not consecutive_missing_counts.empty else 0

        # Impute missing values for the current column using linear interpolation
        teleconnections_filled = teleconnections.copy()
        teleconnections_filled[col] = teleconnections[col].interpolate(method='linear')

        # Count how many consecutive rows are missing from the end in the current column
        last_streak = teleconnections[col].isnull().iloc[::-1].cumsum().max()

        # If there's a streak of missing values at the end, revert those rows back to NaN
        if last_streak > 0:
            last_indices = teleconnections_filled.index[-end_missing_streak:]
            teleconnections_filled.loc[last_indices, col] = np.nan

        # Update the original DataFrame with the filled values
        teleconnections[col] = teleconnections_filled[col]

    return teleconnections


def scale_data(teleconnections):
    """
    Scale the teleconnections dataset using StandardScaler based on the first 804 rows.

    Parameters:
        teleconnections (pd.DataFrame): The teleconnections dataset.

    Returns:
        pd.DataFrame: The scaled teleconnections dataset.
    """
    try:
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler on the first 804 rows
        scaler.fit(teleconnections.iloc[:804])

        # Transform the entire dataset using the fitted scaler
        df_scaled = scaler.transform(teleconnections)

        return pd.DataFrame(df_scaled, index=teleconnections.index, columns=teleconnections.columns)
    except Exception as e:
        print(f"Error in scaling data: {e}")
        return teleconnections
    

def interpolate_missing(teleconnections):
    """
    Function to impute past values of teleconnections data.
    """
    # Iterate through each column in the dataframe
    for col in teleconnections.columns:
    
        # Create a mask for missing values in the current column
        missing_mask = teleconnections[col].isnull()
        group_id = (missing_mask != missing_mask.shift()).cumsum()
    
        # Filter out the groups that are not missing and count consecutive missing values
        consecutive_missing_counts = (
            teleconnections[missing_mask]
            .assign(group_id=group_id[missing_mask])
            .groupby('group_id')
            .size()
        )
    
        # Convert the counts to a DataFrame for better readability
        consecutive_missing_counts = consecutive_missing_counts.reset_index(name='count')
        end_missing_streak = consecutive_missing_counts.iloc[-1]['count'] if not consecutive_missing_counts.empty else 0
    
        # Impute missing values for the current column using linear interpolation
        teleconnections_filled = teleconnections.copy()
        teleconnections_filled[col] = teleconnections[col].interpolate(method='linear')
    
        # Count how many consecutive rows are missing from the end in the current column
        last_streak = teleconnections[col].isnull().iloc[::-1].cumsum().max()
    
        # If there's a streak of missing values at the end, revert those rows back to NaN
        if last_streak > 0:
            last_indices = teleconnections_filled.index[-end_missing_streak:]
            teleconnections_filled.loc[last_indices, col] = np.nan
    
        # Update the original DataFrame with the filled values
        teleconnections[col] = teleconnections_filled[col]
        
    return teleconnections


# Define main function to process teleconnections data
def update_teleconnections():
    """
    Main function to process the teleconnections data from download to final output.
    """
    # Process each teleconnection data source
    df_oni = oni.process_oni()
    df_sst = sst.process_sst()
    df_mjo = mjo.process_mjo()
    df_soi = soi.process_soi()
    df_dmi = dmi.process_dmi()

    # Create a list of dataframes
    teleconnections_dfs = [df_oni, df_sst, df_mjo, df_soi, df_dmi]

    # Add date columns
    teleconnections_dfs = add_date_columns(teleconnections_dfs)

    # Merge all dataframes
    teleconnections = merge_teleconnections_data(teleconnections_dfs)

    # Align data with the specified dates
    dates_list = cleaning_utils.get_dates_of_interest()
    dates_list = pd.to_datetime(dates_list).sort_values()
    teleconnections = align_teleconnections_with_dates(teleconnections, dates_list)

    # Handle missing values
    teleconnections = interpolate_missing(teleconnections)
    teleconnections = cleaning_utils.impute_missing_values(teleconnections, teleconnections.columns)
    
    # Filter dates to study period
    min_date = pd.to_datetime('2002-07-01')
    teleconnections = teleconnections[teleconnections.index >= min_date]

    # Scale the teleconnections data
    teleconnections = scale_data(teleconnections)

    # Save the processed data
    teleconnections.to_csv('data/historic/teleconnections.csv', index=True)
    print("Teleconnections data processing completed successfully.")