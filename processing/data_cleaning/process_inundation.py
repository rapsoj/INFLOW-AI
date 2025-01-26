# Import system libraries
import os

# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime

# Import geospatial libraries
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask

# Import client libraries
import requests

# Import compression libraries
import h5py

# Import progress bar libraries
from tqdm import tqdm

# Import cleaning utils
from .. import cleaning_utils

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_stats(stats_file_path):
    """
    Read the rainfall statistics file.

    Parameters:
        stats_file_path (str): Path to rainfall statistics file.
    """
    with open(stats_file_path, 'r') as f:
        lines = f.readlines()
    # Extract mean and std from the file
    rainfall_mean = float(lines[0].split(': ')[1].strip())
    rainfall_std = float(lines[1].split(': ')[1].strip())
    return rainfall_mean, rainfall_std
    

def download_inundation(dates_list, download_path='data/downloads/inundation_masks'):
    """
    Download inundation data for the specified dates.

    Parameters:
        dates_list (list): List of dates for which to download inundation data.
        download_path (str): Directory path to save downloaded TIF files.
    """
    base_url = "https://data.earthobservation.vam.wfp.org/public-share/sudd_dashboard/ssdmask/ssdmask"
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for date in tqdm(dates_list, desc="Downloading inundation data"):
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        formatted_date = date.strftime('%Y%m%d')
        file_name = f"{formatted_date}.tif"
        file_url = f"{base_url}{file_name}"
        file_path = os.path.join(download_path, file_name)

        try:
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Error occurred while downloading {file_name}: {e}")

def get_sorted_tif_files(folder_path):
    """
    Get a sorted list of TIF files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing TIF files.

    Returns:
        list: Sorted list of TIF file names.
    """
    try:
        tif_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
        tif_files.sort()
        return tif_files
    except FileNotFoundError:
        logging.error(f"Folder not found: {folder_path}")
        return []

def load_shapefile(path):
    """
    Load a shapefile as a GeoDataFrame.

    Parameters:
        path (str): Path to the shapefile.

    Returns:
        GeoDataFrame: Loaded shapefile as a GeoDataFrame.
    """
    try:
        return gpd.read_file(path)
    except FileNotFoundError:
        logging.error(f"Shapefile not found: {path}")
        return None

def reproject_to_raster_crs(shapefile, raster_path):
    """
    Reproject a GeoDataFrame to match the CRS of a raster file.

    Parameters:
        shapefile (GeoDataFrame): The GeoDataFrame to reproject.
        raster_path (str): Path to a raster file for CRS reference.

    Returns:
        GeoDataFrame: Reprojected GeoDataFrame.
    """
    try:
        with rasterio.open(raster_path) as src:
            return shapefile.to_crs(src.crs)
    except Exception as e:
        logging.error(f"Error in reprojecting shapefile: {e}")
        return None

def process_and_clip_rasters(tif_files, folder_path, catchments):
    """
    Clip and collect metadata for each raster file in a folder.

    Parameters:
        tif_files (list): List of TIF file names.
        folder_path (str): Path to the folder containing TIF files.
        catchments (GeoDataFrame): GeoDataFrame of the catchment areas for clipping.

    Returns:
        tuple: Arrays of clipped rasters, list of file names, and metadata dictionary.
    """
    clipped_tif_files = []
    tif_file_names = []
    spatial_metadata = {}

    for file_name in tqdm(tif_files, desc="Processing TIF files"):
        file_path = os.path.join(folder_path, file_name)

        try:
            with rasterio.open(file_path) as src:
                clipped, clipped_transform = rasterio_mask(src, catchments.geometry, crop=True)
                clipped_meta = src.meta.copy()
                clipped_meta.update({
                    "driver": "GTiff",
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": clipped_transform
                })

                clipped_tif_files.append(clipped[0])
                tif_file_names.append(file_name)
                spatial_metadata[file_name] = {
                    "crs": src.crs,
                    "transform": clipped_transform,
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "bounds": src.bounds
                }
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")
            continue

    return clipped_tif_files, tif_file_names, spatial_metadata

def get_historic_dates(data_path='data/historic/inundation_temporal.csv'):
    """
    Get list of historic dates from pre-downloaded data.

    Parameters:
        data_path (str): Directory path of pre-downloaded temporal data.
    """
    try:
        inundation_temporal = pd.read_csv(data_path, index_col=0)
        historic_dates = inundation_temporal.index.tolist()
        return historic_dates
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        return []

def download_new_inundation(download_path='data/downloads/inundation_masks'):
    """
    Download inundation data for dates not already downloaded.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    historic_dates = get_historic_dates()

    if historic_dates:
        last_date = historic_dates[-1]  # Get the last downloaded date
        last_date = datetime.strptime(last_date, "%Y-%m-%d").strftime("%Y-%m-%d")  # Ensure the format is YYYY-MM-DD
    else:
        last_date = "1900-01-01"   # Default value if no dates are found

    new_dates = cleaning_utils.get_dates_of_interest(start_date_str=last_date, end_date_str=current_date_str)

    if new_dates:
        download_inundation(new_dates, download_path)
    else:
        logging.info("No new dates to download.")

def update_inundation(download_path='data/downloads/inundation_masks',
                      temporal_data_path='data/historic/inundation_temporal_unscaled.csv',
                      stats_file_path='data/stats/inundation_stats.txt'):
    """
    Process newly downloaded inundation data and combine it with existing data.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        temporal_data_path (str): Directory path of pre-downloaded temporal data.
    """
    try:
        with h5py.File('data/historic/inundation.h5', 'r') as f:
            inundation_historic = f['inundation']
            logging.info(f"Existing inundation data shape: {inundation_historic.shape}")

        # Update inundation data by downloading new files
        download_new_inundation(download_path)

        # Get the sorted TIF files after the update
        sorted_files = get_sorted_tif_files(download_path)
        historic_dates = get_historic_dates()

        # Identify new files to process (files not already processed)
        file_dates = [datetime.strptime(f.split('.')[0], "%Y%m%d").strftime("%Y-%m-%d") for f in sorted_files]
        new_dates_indices = [i for i, date in enumerate(file_dates) if date not in historic_dates]
        new_files = [sorted_files[i] for i in new_dates_indices]

        if not new_files:
            logging.info("No new files to process.")
            return

        # Process the new TIF files
        catchments_path = "data/maps/INFLOW_cmts_15/INFLOW_all_cmts.shp"
        catchments = load_shapefile(catchments_path)
        first_raster_path = os.path.join(download_path, sorted_files[0])
        catchments = reproject_to_raster_crs(catchments, first_raster_path)

        # Process rasters and gather new data
        new_clipped_tif_files, _, _ = process_and_clip_rasters(new_files, download_path, catchments)
        
                # Calculate total number of cells
        total_cells = new_clipped_tif_files[0].shape[0] * new_clipped_tif_files[0].shape[1]
        
        # Create new temporal data
        inundation_temporal = pd.DataFrame(np.sum(new_clipped_tif_files, axis=(1, 2)) / total_cells, columns=["percent_inundation"])
        
        # Fix index creation by looping over new_files
        inundation_temporal['date'] = [datetime.strptime(file.split('.')[0], "%Y%m%d").date() for file in new_files]
        
        # Update temporal data
        inundation_temporal_historic = pd.read_csv(temporal_data_path)
        inundation_temporal_new = pd.concat([inundation_temporal_historic, inundation_temporal])
        temporal_mean, temporal_std = read_stats(stats_file_path)
        inundation_temporal_new_scaled = inundation_temporal_new.copy()
        inundation_temporal_new_scaled['percent_inundation'] = (inundation_temporal_new_scaled['percent_inundation'] - temporal_mean) / temporal_std
        
        # Save the updated temporal data
        inundation_temporal_new.to_csv('data/historic/inundation_temporal_unscaled.csv', index=False)
        inundation_temporal_new_scaled.to_csv('data/historic/inundation_temporal_scaled.csv', index=False)

        # Combine existing and new inundation data
        with h5py.File('data/historic/inundation.h5', 'a') as hdf:
            dset = hdf['inundation']
            dset.resize(dset.shape[0] + len(new_clipped_tif_files), axis=0)
            dset[-len(new_clipped_tif_files):] = new_clipped_tif_files
            logging.info(f"Updated inundation data shape: {dset.shape}")
            
    except Exception as e:
        logging.error(f"Error processing new inundation data: {e}")

    logging.info("Inundation processing complete.\n")