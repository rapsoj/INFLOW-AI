import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from functools import reduce
from download_teleconnections import oni, sst, mjo, soi, dmi  # Assuming each teleconnection module has a process function
from utils import get_dates_interest, impute_missing_values  # Assuming these functions are in a utils module

# Step 1: Process teleconnections data
def load_teleconnections():
    df_oni = oni.process_oni()
    df_sst = sst.process_sst()
    df_mjo = mjo.process_mjo()
    df_soi = soi.process_soi()
    df_dmi = dmi.process_dmi()
    return [df_oni, df_sst, df_mjo, df_soi, df_dmi]

# Step 2: Format teleconnection dates
def format_teleconnection_dates(teleconnections_dfs):
    for i, df in enumerate(teleconnections_dfs):
        if 'day' not in df.columns:
            df['day'] = 25
            df['date_represented'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str))
            df['date_reported'] = df['date_represented'] + pd.DateOffset(months=1)
        else:
            df['date_represented'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str))
            df['date_reported'] = df['date_represented'] + pd.DateOffset(days=5)
        teleconnections_dfs[i] = df.drop(['year', 'month', 'day'], axis=1)
    return teleconnections_dfs

# Step 3: Combine teleconnections dataframes on 'date_represented' and 'date_reported'
def combine_teleconnections(teleconnections_dfs):
    return reduce(lambda left, right: pd.merge(left, right, on=['date_represented', 'date_reported'], how='outer'), teleconnections_dfs)

# Step 4: Find nearest future date
def find_nearest_future_date(date, date_list):
    future_dates = [d for d in date_list if d < date]
    return future_dates[-1] if future_dates else np.nan

# Step 5: Align teleconnection data with dates of interest
def align_with_dates(teleconnections, dates_list):
    teleconnections['date'] = teleconnections['date_represented'].apply(lambda x: find_nearest_future_date(x, dates_list))
    teleconnections = teleconnections.groupby('date').mean().reset_index()
    
    date_df = pd.DataFrame({'date': pd.to_datetime(dates_list).sort_values()})
    teleconnections = pd.merge(date_df, teleconnections, on='date', how='left')
    
    return teleconnections.set_index('date').drop(['date_represented', 'date_reported'], axis=1)

# Step 6: Interpolate missing values with constraints on end-of-data streaks
def interpolate_missing_values(df):
    df_filled = df.copy()
    for col in df.columns:
        missing_mask = df[col].isnull()
        group_id = (missing_mask != missing_mask.shift()).cumsum()
        
        consecutive_missing_counts = (df[missing_mask]
                                      .assign(group_id=group_id[missing_mask])
                                      .groupby('group_id')
                                      .size()
                                      .reset_index(name='count'))
        end_missing_streak = consecutive_missing_counts.iloc[-1]['count'] if not consecutive_missing_counts.empty else 0
        
        df_filled[col] = df[col].interpolate(method='linear')
        last_streak = df[col].isnull().iloc[::-1].cumsum().max()
        
        if last_streak > 0:
            last_indices = df_filled.index[-end_missing_streak:]
            df_filled.loc[last_indices, col] = np.nan

        df[col] = df_filled[col]
    return df

# Step 7: Filter data to study period
def filter_study_period(df, start_date='2002-07-01'):
    return df[df.index >= pd.to_datetime(start_date)]

# Step 8: Standardize teleconnections data
def standardize_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, index=df.index, columns=df.columns)

# Main processing function
def process_teleconnections():
    # Load and format teleconnections data
    teleconnections_dfs = load_teleconnections()
    teleconnections_dfs = format_teleconnection_dates(teleconnections_dfs)

    # Combine teleconnections and align with dates of interest
    teleconnections = combine_teleconnections(teleconnections_dfs)
    dates_list = get_dates_interest()
    teleconnections = align_with_dates(teleconnections, dates_list)

    # Interpolate missing values, filter study period, and standardize data
    teleconnections = interpolate_missing_values(teleconnections)
    teleconnections = filter_study_period(teleconnections)
    teleconnections = standardize_data(teleconnections)

    # Impute any remaining missing values using regression or other techniques
    columns_to_impute = teleconnections.columns.tolist()
    teleconnections = impute_missing_values(teleconnections, columns_to_impute, regression_length=6)

    # Save the processed data
    teleconnections.to_csv('outputs/teleconnections.csv', index=True)
    print("Teleconnections data processed and saved.")