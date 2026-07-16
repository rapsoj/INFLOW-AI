import os
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from processing import cleaning_utils
from inflow_ai.settings import get_settings


SETTINGS = get_settings()


def create_dataframe():
    """
    Create dataframe from recently refreshed data.
    """
    # Load data
    victoria = pd.read_csv(f'{SETTINGS.historic_dir}/victoria.csv', index_col='date')
    albert = pd.read_csv(f'{SETTINGS.historic_dir}/albert.csv', index_col='date')
    kyoga = pd.read_csv(f'{SETTINGS.historic_dir}/kyoga.csv', index_col='date')
    rainfall = pd.read_csv(f'{SETTINGS.historic_dir}/rainfall.csv', index_col='date')
    teleconnections = pd.read_csv(f'{SETTINGS.historic_dir}/teleconnections.csv', index_col='date')
    inundation_temporal_scaled = pd.read_csv(SETTINGS.inundation_temporal_scaled, index_col='date')
    gridded_rainfall_temporal = pd.read_csv(f'{SETTINGS.historic_dir}/gridded_rainfall_temporal.csv', index_col='date')
    gridded_rainfall_cumulative_temporal = pd.read_csv(f'{SETTINGS.historic_dir}/gridded_rainfall_cumulative_temporal.csv', index_col='date')
    gridded_moisture_temporal = pd.read_csv(f'{SETTINGS.historic_dir}/gridded_moisture_temporal.csv', index_col='date')

    # Calculate inundation delta
    inundation_temporal_unscaled = pd.read_csv(SETTINGS.inundation_temporal_unscaled, index_col='date')
    inundation_temporal_delta = inundation_temporal_unscaled[['percent_inundation']].diff()
    inundation_temporal_delta.columns = ['inundation_delta']

    # Combine data into temporal dataframe
    temporal_data_df = pd.concat([
        victoria,
        albert,
        kyoga,
        rainfall,
        teleconnections,
        inundation_temporal_scaled.rename({'percent_inundation': 'inundation_temporal'}, axis=1)[['inundation_temporal']],
        gridded_rainfall_temporal.rename({'rainfall': 'rainfall_3d_temporal'}, axis=1)[['rainfall_3d_temporal']],
        gridded_rainfall_cumulative_temporal.rename({'cumulative_rainfall': 'rainfall_cumulative'}, axis=1)[['rainfall_cumulative']],
        gridded_moisture_temporal.rename({'moisture': 'moisture_3d_temporal'}, axis=1)[['moisture_3d_temporal']],
        inundation_temporal_delta
    ], axis=1)
    
    temporal_data_df = cleaning_utils.impute_missing_values(temporal_data_df, temporal_data_df.drop(columns=['inundation_temporal', 'inundation_delta']).columns)
    
    return temporal_data_df


def export_graphs_all_vars(data, explainer_vars):
    """
    Plot year-by-year comparison graphs for specified variables and save them into:
    /predictions/<most_recent_subfolder_by_name_date_or_ctime>/explanations/

    Accepts data with either a datetime index or a "date" column containing "YYYY-MM-DD" strings.
    """
    # --- prepare dataframe and datetime index ---
    df_all = data.copy()
    if "date" in df_all.columns:
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        df_all = df_all.dropna(subset=["date"])
        df_all = df_all.set_index("date").sort_index()
    else:
        # try convert index to datetime
        try:
            if not isinstance(df_all.index, pd.DatetimeIndex):
                df_all.index = pd.to_datetime(df_all.index, errors="coerce")
        except Exception:
            df_all.index = pd.to_datetime(df_all.index, errors="coerce")

    if not isinstance(df_all.index, pd.DatetimeIndex):
        raise ValueError("Could not convert dates to a DatetimeIndex. Provide a 'date' column or datetime-like index.")
    if df_all.empty:
        raise ValueError("Dataframe is empty after date conversion/cleaning.")

    # --- locate most recent subfolder inside /predictions by folder name date or fallback to ctime ---
    predictions_root = SETTINGS.predictions_dir
    if not os.path.isdir(predictions_root):
        raise FileNotFoundError("No 'predictions' folder found in working directory.")

    subdirs = [os.path.join(predictions_root, p) for p in os.listdir(predictions_root)
               if os.path.isdir(os.path.join(predictions_root, p))]
    if not subdirs:
        raise FileNotFoundError("No subfolders found inside /predictions.")

    # helper: try to extract YYYY-MM-DD or YYYYMMDD from folder name
    def parse_date_from_name(name):
        # look for ISO-like YYYY-MM-DD
        m = re.search(r'(\d{4}-\d{2}-\d{2})', name)
        if m:
            try:
                return datetime.fromisoformat(m.group(1)).date()
            except Exception:
                pass
        # look for contiguous YYYYMMDD
        m2 = re.search(r'(\d{8})', name)
        if m2:
            s = m2.group(1)
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                pass
        return None

    parsed = []
    for p in subdirs:
        basename = os.path.basename(p)
        parsed_date = parse_date_from_name(basename)
        parsed.append((p, parsed_date, os.path.getctime(p)))

    # If at least one folder had a parseable date, choose the one with the max parsed_date.
    parsed_with_dates = [t for t in parsed if t[1] is not None]
    if parsed_with_dates:
        chosen_path = max(parsed_with_dates, key=lambda t: t[1])[0]
    else:
        # fallback: choose by creation time
        chosen_path = max(parsed, key=lambda t: t[2])[0]

    explanations_folder = os.path.join(chosen_path, "explanations")
    os.makedirs(explanations_folder, exist_ok=True)

    # --- plotting setup ---
    start_label = df_all.index.min().date().strftime('%Y-%m-%d')
    end_label = df_all.index.max().date().strftime('%Y-%m-%d')
    cmap = plt.get_cmap('winter')

    for var in tqdm(explainer_vars, "Plotting explanations"):
        if var not in df_all.columns:
            print(f"Skipping '{var}': not in dataframe.")
            continue

        series = df_all[var].dropna().astype(float)
        if series.empty:
            print(f"Skipping '{var}': no non-NaN values after cleaning.")
            continue

        plot_df = pd.DataFrame({'value': series})
        plot_df['Year'] = plot_df.index.year
        plot_df['DayOfYear'] = plot_df.index.dayofyear

        fig, ax = plt.subplots(figsize=(10, 6))
        unique_years = sorted(plot_df['Year'].unique())

        for i, (year, group) in enumerate(plot_df.groupby('Year')):
            group = group.sort_index()
            if group.empty:
                continue

            if year == datetime.today().year:
                ax.plot(group['DayOfYear'], group['value'], color='red', linewidth=2)
                text_color = 'black'
            else:
                color = cmap(i / max(1, len(unique_years)))
                ax.plot(group['DayOfYear'], group['value'], color=color, alpha=0.6)
                text_color = color

            ax.text(group['DayOfYear'].iloc[-1],
                    group['value'].iloc[-1],
                    str(year),
                    color=text_color,
                    fontsize=9,
                    va='center')

        ax.set_title(f"{var}: {start_label} to {end_label} ({datetime.today().year} highlighted)")
        ax.set_xlabel('Month of Year')
        ax.set_ylabel(var)

        months = pd.date_range(start=f"{datetime.today().year}-01-01", periods=12, freq='MS')
        month_days = [d.dayofyear for d in months]
        ax.set_xticks(month_days)
        ax.set_xticklabels([d.strftime('%b') for d in months])
        ax.set_xlim(1, 366)

        plt.tight_layout()
        outname = f"{var}_{start_label}_to_{end_label}_year_by_year_comparison.png"
        plt.savefig(os.path.join(explanations_folder, outname), dpi=300)
        plt.close()

def get_explanations():
    df = create_dataframe()
    export_graphs_all_vars(df, list(df.columns))