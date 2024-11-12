import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from tqdm import tqdm
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import reproject, Resampling
import cupy as cp
from processing import get_dates_interest

# Constants for file paths and folders
INUNDATION_FOLDER = "inundation_masks_updated"
RAINFALL_FOLDER = "tamsat/rainfall_tifs_updated"
CATCHMENTS_PATH = "Project Map and Shapefiles/INFLOW_cmts_clean/INFLOW_cmts_15/INFLOW_all_cmts.shp"
OUTPUT_FOLDER = "outputs"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Identify TIF files in a folder
def identify_tif_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith(".tif")]

# Function to load and check CRS match for catchments with a sample tif file
def load_and_match_crs(catchments_path, sample_tif_path):
    catchments = gpd.read_file(catchments_path)
    with rasterio.open(sample_tif_path) as src:
        catchments = catchments.to_crs(src.crs)
    return catchments

# Function to get transform, dimensions, CRS, and resolution of a sample TIF file
def get_sample_tif_properties(sample_tif_path, catchments):
    with rasterio.open(sample_tif_path) as src:
        clipped, transform = rasterio_mask(src, catchments.geometry, crop=True)
        return {
            "transform": transform,
            "width": clipped.shape[2],
            "height": clipped.shape[1],
            "crs": src.crs,
            "resolution": src.res
        }

# Reproject and resample rainfall data to match sample properties
def reproject_rainfall(rainfall_ds, target_crs, target_transform, target_width, target_height):
    rainfall_data = rainfall_ds.read(1)  # Read the first band
    reprojected_rainfall = np.empty((target_height, target_width), dtype=rainfall_data.dtype)
    reproject(
        source=rainfall_data,
        destination=reprojected_rainfall,
        src_transform=rainfall_ds.transform,
        src_crs=rainfall_ds.crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )
    return reprojected_rainfall

# Function to extract date from TIF filename
def extract_date_from_filename(filename):
    if filename.endswith('.tif'):
        date_str = ''.join(filter(str.isdigit, filename.split('_')[-1].split('.')[0]))
        try:
            return pd.to_datetime(date_str, format='%Y%m%d')
        except ValueError:
            return None
    return None

# Align rainfall files to MODIS dates of interest
def align_rainfall_files(rainfall_folder, dates_list):
    rainfall_files = [f for f in os.listdir(rainfall_folder) if f.endswith('.tif')]
    rainfall_dates = [extract_date_from_filename(f) for f in rainfall_files]
    rainfall_df = pd.DataFrame({
        "rainfall_file": rainfall_files,
        "rainfall_date": rainfall_dates
    }).dropna().sort_values("rainfall_date")
    
    dates_df = pd.DataFrame({"date": pd.to_datetime(dates_list)})
    aligned_df = pd.merge(dates_df, rainfall_df, left_on="date", right_on="rainfall_date", how="left")

    missing_dates = aligned_df[aligned_df['rainfall_file'].isna()]['date']
    if not missing_dates.empty:
        print(f"Warning: Missing rainfall data for {len(missing_dates)} dates.")
    
    return aligned_df['rainfall_file'].tolist()

# Function to process rainfall data into a 3D array with CuPy
def process_rainfall_data_to_array(aligned_rainfall_files, rainfall_folder, sample_props):
    rainfall_data_list = []
    for rainfall_tif in tqdm(aligned_rainfall_files, desc="Processing aligned rainfall TIF files"):
        if pd.notna(rainfall_tif):
            rainfall_tif_path = os.path.join(rainfall_folder, rainfall_tif)
            with rasterio.open(rainfall_tif_path) as rainfall_ds:
                resampled_rainfall = reproject_rainfall(
                    rainfall_ds,
                    target_crs=sample_props["crs"],
                    target_transform=sample_props["transform"],
                    target_width=sample_props["width"],
                    target_height=sample_props["height"]
                )
            rainfall_data_list.append(cp.asarray(resampled_rainfall))
    return cp.stack(rainfall_data_list, axis=0)

# Replace missing values in the 3D array
def replace_missing_values(rainfall_3d_array, fill_value=1.000000000000000e+5):
    rainfall_3d_array[rainfall_3d_array >= fill_value] = 0
    return rainfall_3d_array

# Calculate mean and standard deviation across the 3D array
def calculate_statistics(rainfall_3d_array):
    time, rows, cols = rainfall_3d_array.shape
    total_sum = cp.sum(rainfall_3d_array)
    total_sum_sq = cp.sum(rainfall_3d_array ** 2)
    count = time * rows * cols

    mean = total_sum / count
    std_dev = cp.sqrt(total_sum_sq / count - mean ** 2)

    with open(os.path.join(OUTPUT_FOLDER, 'rainfall_stats.txt'), 'w') as f:
        f.write(f"Rainfall Mean: {mean}\n")
        f.write(f"Rainfall Std Dev: {std_dev}\n")
    
    return mean, std_dev

# Standardize the 3D array based on calculated mean and standard deviation
def standardize_rainfall_data(rainfall_3d_array, mean, std_dev):
    return (rainfall_3d_array - mean) / std_dev

# Main function to process rainfall gridded data
def process_rainfall_gridded():
    tif_file_names = identify_tif_files(INUNDATION_FOLDER)
    modis_dates = pd.to_datetime([i[:8] for i in tif_file_names])

    # Sample properties for reprojection and resampling
    sample_tif_path = os.path.join(INUNDATION_FOLDER, tif_file_names[0])
    catchments = load_and_match_crs(CATCHMENTS_PATH, sample_tif_path)
    sample_props = get_sample_tif_properties(sample_tif_path, catchments)

    # Align rainfall files
    rainfall_dates = get_dates_interest(start_date_str=str(modis_dates.min())[:-9])
    aligned_rainfall_files = align_rainfall_files(RAINFALL_FOLDER, rainfall_dates)

    # Convert rainfall data to a 3D array
    rainfall_3d_array = process_rainfall_data_to_array(aligned_rainfall_files, RAINFALL_FOLDER, sample_props)

    # Handle missing values
    rainfall_3d_array = replace_missing_values(rainfall_3d_array)

    # Calculate mean and standard deviation
    rainfall_mean, rainfall_std = calculate_statistics(rainfall_3d_array)

    # Standardize rainfall data
    rainfall_3d_array = standardize_rainfall_data(rainfall_3d_array, rainfall_mean, rainfall_std)

    # Crop to dates of interest and save
    first_date_id = rainfall_dates.index('2002-07-01')
    rainfall_3d_array_cropped = rainfall_3d_array[first_date_id:, :, :]
    cp.save(os.path.join(OUTPUT_FOLDER, 'rainfall_3d_array_cropped.npy'), rainfall_3d_array_cropped)
    cp.save(os.path.join(OUTPUT_FOLDER, 'rainfall_3d_array.npy'), rainfall_3d_array)

    print("Rainfall gridded data processing completed.")