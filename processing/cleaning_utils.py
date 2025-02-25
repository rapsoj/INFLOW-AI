# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import machine learning libraries
from sklearn.linear_model import LinearRegression

def get_dates_of_interest(start_date_str='2002-07-01', end_date_str=None):
    """
    Generate a list of dates between start_date_str and end_date_str where the day ends in '01', '11', or '21'.

    Parameters:
        start_date_str (str): The start date in 'YYYY-MM-DD' format. Defaults to '2002-07-01'.
        end_date_str (str): The end date in 'YYYY-MM-DD' format. Defaults to 60 days from today if not provided.

    Returns:
        list: A list of dates (in 'YYYY-MM-DD' format) where the day is 1, 11, or 21.
    """
    # Parse the start date
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid start_date_str format. Expected 'YYYY-MM-DD', got: {start_date_str}")
    
    # Get today's date if end_date_str is not provided, otherwise parse the end_date_str
    if not end_date_str:
        end_date = datetime.today()
    else:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid end_date_str format. Expected 'YYYY-MM-DD', got: {end_date_str}")
    
    # Initialize an empty list to store the dates
    dates_of_interest = []

    # Iterate through all dates between start_date and end_date
    current_date = start_date
    while current_date <= end_date:
        # Check if the day ends in '01', '11', or '21'
        if current_date.day in [1, 11, 21]:
            dates_of_interest.append(current_date.strftime('%Y-%m-%d'))
        # Move to the next day
        current_date += timedelta(days=1)

    return dates_of_interest

# Define function to linearly extrapolate missing values between data points
def impute_missing_values(df, cols, regression_length=6):
    """
    Impute missing values for each column in the provided dataframe using linear regression.

    Parameters:
    - df: pd.DataFrame
        The dataframe containing time series data with missing values.
    - cols: list
        A list of column names to impute using linear regression.
    - regression_length: int, default=6
        The number of past non-missing data points to use for linear regression.

    Returns:
    - df_imputed: pd.DataFrame
        The dataframe with missing values imputed for the specified columns.
    """
    # Get the current date
    current_date = pd.Timestamp.now()

    # Filter dates that are before the current date
    original_index = df.index
    df.index = pd.to_datetime(df.index)
    dates = df.index
    past_dates = dates[dates <= current_date]
    data = df.loc[past_dates]

    # Loop through each column to impute missing values
    for col in cols:
        # Split the data for the current column
        past_data = data[data[col].notna()][-regression_length:]  # Last non-missing values
        impute_data = data[data[col].isna()]  # Data with NaNs to impute
        forecast_steps = len(impute_data)

        if forecast_steps == 0 or len(past_data) < regression_length:
            # If there are no missing values to impute or not enough past data, skip this column
            continue

        # Prepare data for linear regression
        print(f"Imputing {forecast_steps} timestep for {col}.")
        series = past_data[col]
        X = np.arange(len(series)).reshape(-1, 1)  # Time index (0, 1, ..., n)
        y = series.values  # Corresponding values

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Create future indices based on the missing data indices
        future_indices = impute_data.index  # Get indices for missing dates

        # Generate future time indices for predictions (match length of forecast steps)
        future_time_steps = np.arange(len(series), len(series) + forecast_steps).reshape(-1, 1)
        forecast_values = model.predict(future_time_steps)

        # Create a Series for the forecast values (imputed points)
        forecast_df = pd.Series(forecast_values, index=future_indices, name=f'{col}_Forecast')

        # Combine the forecasted values with the original data
        df.loc[future_indices, col] = forecast_df
    
    # Reset index
    df.index = original_index

    return df