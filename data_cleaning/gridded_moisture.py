import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

# Paths and configurations
MOISTURE_FOLDER = 'tamsat/moisture_tifs_updated'
TARGET_FOLDER = "inundation_masks_updated"
CATCHMENTS_PATH = "Project Map and Shapefiles/INFLOW_cmts_clean/INFLOW_cmts_15/INFLOW_all_cmts.shp"
OUTPUT_MOISTURE_ARRAY_PATH = 'outputs/moisture_3d_array.npy'
STATS_FILE_PATH = 'moisture_stats.txt'

# Helper function to get sample TIF file for CRS and dimensions reference
def get_sample_tif_path(folder):
    return os.path.join(folder, os.listdir(folder)[0])

# Ensure catchments CRS matches a given raster's CRS
def ensure_crs_match(geodf, raster_file):
    with rasterio.open(raster_file) as src:
        return geodf.to_crs(src.crs)

# Function to read sample TIF for transform and dimensions
def get_sample_tif_properties(tif_path, catchments):
    with rasterio.open(tif_path) as src:
        clipped, clipped_transform = rasterio_mask(src, catchments.geometry, crop=True)
        return {
            "width": clipped.shape[2],
            "height": clipped.shape[1],
            "crs": src.crs,
            "transform": clipped_transform,
            "res": src.res
        }

# Reproject and resample a moisture file
def reproject_moisture(moisture_ds, target_crs, target_transform, target_width, target_height):
    moisture_data = moisture_ds.read(1)
    reprojected_moisture = np.empty((target_height, target_width), dtype=moisture_data.dtype)
    reproject(
        source=moisture_data,
        destination=reprojected_moisture,
        src_transform=moisture_ds.transform,
        src_crs=moisture_ds.crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )
    return reprojected_moisture

# Extract date from filename
def extract_date_from_filename(filename):
    if filename.endswith('.tif'):
        date_str = ''.join(filter(str.isdigit, filename.split('_')[-1].split('.')[0]))
        try:
            return pd.to_datetime(date_str, format='%Y%m%d')
        except ValueError:
            return None
    return None

# Align moisture files by date
def align_moisture_files(dates, moisture_files):
    moisture_dates = [extract_date_from_filename(f) for f in moisture_files]
    moisture_df = pd.DataFrame({'moisture_file': moisture_files, 'moisture_date': moisture_dates})
    aligned_df = pd.merge(pd.DataFrame({'date': dates}), moisture_df, left_on='date', right_on='moisture_date', how='left')
    return aligned_df['moisture_file'].tolist()

# Process moisture data files into 3D array
def process_moisture_data_files(moisture_files, sample_properties):
    moisture_data_list = []
    for moisture_tif in tqdm(moisture_files, desc="Processing moisture TIF files"):
        if pd.notna(moisture_tif):
            with rasterio.open(os.path.join(MOISTURE_FOLDER, moisture_tif)) as moisture_ds:
                resampled_moisture = reproject_moisture(
                    moisture_ds,
                    target_crs=sample_properties["crs"],
                    target_transform=sample_properties["transform"],
                    target_width=sample_properties["width"],
                    target_height=sample_properties["height"]
                )
                moisture_data_list.append(resampled_moisture)
    return np.stack(moisture_data_list, axis=0)

# Calculate and save global statistics
def calculate_statistics(array, stats_path=STATS_FILE_PATH):
    time, rows, cols = array.shape
    total_sum = np.sum(array)
    total_sum_sq = np.sum(array ** 2)
    count = time * rows * cols

    mean = total_sum / count
    std_dev = np.sqrt(total_sum_sq / count - mean ** 2)

    with open(stats_path, 'w') as f:
        f.write(f"Moisture Mean: {mean}\n")
        f.write(f"Moisture Std Dev: {std_dev}\n")
    
    return mean, std_dev

# Apply standard scaling to the moisture array
def standardize_array(array, mean, std_dev):
    return (array - mean) / std_dev

# Main function to process the gridded moisture data
def process_gridded_moisture():
    # Load dates and catchments
    dates = pd.to_datetime(get_dates_interest())
    sample_tif_path = get_sample_tif_path(TARGET_FOLDER)
    catchments = gpd.read_file(CATCHMENTS_PATH)
    catchments = ensure_crs_match(catchments, sample_tif_path)

    # Get sample TIF properties
    sample_properties = get_sample_tif_properties(sample_tif_path, catchments)

    # Align and sort moisture files
    moisture_files = [f for f in os.listdir(MOISTURE_FOLDER) if f.endswith('.tif') and not f.endswith('(1).tif')]
    aligned_moisture_files = align_moisture_files(dates, sorted(moisture_files))

    # Process aligned moisture files
    moisture_3d_array = process_moisture_data_files(aligned_moisture_files, sample_properties)

    # Calculate and save statistics
    mean, std_dev = calculate_statistics(moisture_3d_array)

    # Standardize and save moisture data
    moisture_3d_array = standardize_array(moisture_3d_array, mean, std_dev)
    np.save(OUTPUT_MOISTURE_ARRAY_PATH, moisture_3d_array)

    print("All moisture data processed and saved.")
    print(f"Moisture 3D array shape: {moisture_3d_array.shape}")