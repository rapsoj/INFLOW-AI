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

# Import statistics
from data.stats import gridded_data_stats

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_stats(region='all'):
    """
    Read the gridded data statistics file.
    """
    inundation_mean = gridded_data_stats.inundation_stats[region]['mean']
    inundation_std = gridded_data_stats.inundation_stats[region]['std']
    
    return inundation_mean, inundation_std
    

def download_inundation(dates_list, download_path='data/downloads/inundation_masks'):
    """
    Download inundation data for the specified dates.

    Parameters:
        dates_list (list): List of dates for which to download inundation data.
        download_path (str): Directory path to save downloaded TIF files.
    """
    base_url = "https://data.earthobservation.vam.wfp.org/public-share/sudd_wetland_monitoring/modis_flood_masks/ssdmask"
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


def get_historic_dates(data_path='data/historic/inundation_temporal_unscaled.csv'):
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


def download_new_inundation(download_path='data/downloads/inundation_masks', burn_in_steps=18):
    """
    Download inundation data for the last `burn_in_steps` timesteps (to refresh them)
    plus any new dates up to the current date.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        burn_in_steps (int): Number of timesteps to always refresh.
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    historic_dates = get_historic_dates()

    if historic_dates:
        # Always include the last N dates to refresh
        if len(historic_dates) >= burn_in_steps:
            start_date = historic_dates[-burn_in_steps]
        else:
            start_date = historic_dates[0]
    else:
        # Default if no history exists
        start_date = "2002-07-01"

    # Get all dates of interest (last N + up to today)
    new_dates = cleaning_utils.get_dates_of_interest(
        start_date_str=start_date,
        end_date_str=current_date_str
    )

    if new_dates:
        logging.info(f"Downloading {len(new_dates)} dates (including last {burn_in_steps} for refresh).")
        download_inundation(new_dates, download_path)
    else:
        logging.info("No new dates to download.")
        
        
def crop_historic_data(file_path, temporal_data_path, temporal_data_path_scaled):
    """
    Crop or recreate the historic inundation HDF5 dataset to match the temporal CSV lengths.

    If the HDF5 dataset is longer than the CSVs, the entire HDF5 file will be truncated and recreated.
    Both the original and scaled temporal CSVs are cropped if the HDF5 dataset is shorter.
    """

    # --- Load temporal data lengths only ---
    hist = pd.read_csv(temporal_data_path)
    hist_scaled = pd.read_csv(temporal_data_path_scaled)
    new_len = min(len(hist), len(hist_scaled))

    # --- Open HDF5 and check dataset length ---
    with h5py.File(file_path, "r+") as f:
        dset_name = list(f.keys())[0]
        dset = f[dset_name]
        current_len = dset.shape[0]

        # --- If HDF5 shorter, crop CSVs to match ---
        if current_len < new_len:
            hist.iloc[:current_len].to_csv(temporal_data_path, index=False)
            hist_scaled.iloc[:current_len].to_csv(temporal_data_path_scaled, index=False)
            print(f"✂️ Cropped CSVs to {current_len} timesteps.")

        # --- If HDF5 longer, recreate file (truncate + rewrite) ---
        elif current_len > new_len:
            print(f"Cropping HDF5 from {current_len} → {new_len} timesteps...")

            # Read cropped data before removing file
            data = dset[:new_len]
            dtype, shape = dset.dtype, data.shape
            f.close()  # close handle before removing

            # Remove and recreate (truncate)
            os.remove(file_path)
            with h5py.File(file_path, "w") as newf:
                newf.create_dataset(
                    dset_name,
                    data=data,
                    maxshape=(None, *shape[1:]),
                    chunks=True,
                    dtype=dtype,
                )
            print("✅ HDF5 file truncated and recreated with cropped data.")

        else:
            print("✅ No cropping needed. Temporal lengths already match.")
        
        
def remove_burn_in_data(h5_file_path="data/historic/inundation.h5",
                        temporal_data_path="data/historic/inundation_temporal_unscaled.csv",
                        temporal_data_path_scaled="data/historic/inundation_temporal_scaled.csv",
                        dset_name="inundation",
                        burn_in_steps=18):
    """
    Remove the last `burn_in_steps` dekads from saved MODIS data 
    (spatio-temporal HDF5 dataset and temporal CSVs).
    
    Parameters:
        h5_file_path (str): Path to spatio-temporal historic MODIS HDF5 file.
        temporal_data_path (str): Path to temporal unscaled CSV file.
        temporal_data_path_scaled (str): Path to temporal scaled CSV file.
        dset_name (str): Name of dataset inside the HDF5 file.
        burn_in_steps (int): Number of timesteps (along axis 0) to drop from the end.
    """
    import pandas as pd
    import h5py

    # --- Process HDF5 file ---
    with h5py.File(h5_file_path, "r") as f:
        if dset_name not in f:
            raise KeyError(f"Dataset '{dset_name}' not found in {h5_file_path}.")
        data = f[dset_name][:]

    # Crop last axis (remove last `burn_in_steps` entries)
    if data.shape[0] <= burn_in_steps:
        raise ValueError("Not enough timesteps to remove burn-in data.")
    data_cropped = data[:-burn_in_steps]

    # Overwrite file with cropped dataset
    with h5py.File(h5_file_path, "w") as f:
        dset = f.create_dataset(
            dset_name,
            shape=data_cropped.shape,
            maxshape=(None, *data_cropped.shape[1:]),
            chunks=True,
            dtype=data_cropped.dtype,
        )
        dset[:] = data_cropped

    # --- Process temporal CSV files ---
    for csv_path in [temporal_data_path, temporal_data_path_scaled]:
        df = pd.read_csv(csv_path)
        if len(df) <= burn_in_steps:
            raise ValueError(f"Not enough rows in {csv_path} to remove burn-in data.")
        
        # Remove last burn-in rows
        df_cropped = df.iloc[:-burn_in_steps].reset_index(drop=True)
        
        # Ensure sorted by 'date' column if it exists
        if "date" in df_cropped.columns:
            df_cropped["date"] = pd.to_datetime(df_cropped["date"])
            df_cropped = df_cropped.sort_values("date").reset_index(drop=True)
        
        # Save back to CSV
        df_cropped.to_csv(csv_path, index=False)
    
    print(f"Removed last {burn_in_steps} timesteps from HDF5 and temporal CSVs (sorted by date).")
        

def update_inundation(download_path='data/downloads/inundation_masks',
                      temporal_data_path='data/historic/inundation_temporal_unscaled.csv',
                      temporal_data_path_scaled='data/historic/inundation_temporal_scaled.csv'):
    """
    Process newly downloaded inundation data and combine it with existing data.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        temporal_data_path (str): Directory path of pre-downloaded temporal data.
        temporal_data_path_scaled (str): Directory path of pre-downloaded scaled temporal data.
    """
    try:
        with h5py.File('data/historic/inundation.h5', 'r') as f:
            inundation_historic = f['inundation']
            logging.info(f"Existing inundation data shape: {inundation_historic.shape}")
            
        # Crop historic data if historic spatial and temporal data are not the same size   
        crop_historic_data(
            file_path="data/historic/inundation.h5",
            temporal_data_path=temporal_data_path,
            temporal_data_path_scaled=temporal_data_path_scaled
        )
            
        # Remove burn-in data
        remove_burn_in_data()

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
        catchments_path = "data/maps/inflow_catchments/INFLOW_all_cmts.shp"
        catchments = load_shapefile(catchments_path)
        first_raster_path = os.path.join(download_path, sorted_files[0])
        catchments = reproject_to_raster_crs(catchments, first_raster_path)

        # Process rasters and gather new data
        new_clipped_tif_files, _, _ = process_and_clip_rasters(new_files, download_path, catchments)
        
        # Crop area to regions of interest
        regions_gdf = cleaning_utils.extract_regions()
        
        # Calculate total number of cells
        total_cells = new_clipped_tif_files[0].shape[0] * new_clipped_tif_files[0].shape[1]
        
        # Create new temporal data
        inundation_temporal = pd.DataFrame(np.sum(new_clipped_tif_files, axis=(1, 2)) / total_cells, columns=["percent_inundation"])
        
        # Fix index creation by looping over new_files
        inundation_temporal['date'] = [datetime.strptime(file.split('.')[0], "%Y%m%d").date() for file in new_files]
        inundation_temporal_scaled = inundation_temporal.copy()
        temporal_mean, temporal_std = read_stats()
        inundation_temporal_scaled['percent_inundation'] = (inundation_temporal_scaled['percent_inundation'] - temporal_mean) / temporal_std
        
        # Loop through regions
        for i in range(len(regions_gdf)):
            region_data = regions_gdf.iloc[[i]]
            region_code = gridded_data_stats.region_to_code_dict[region_data['region'].values[0]]
            region_area = cleaning_utils.mask_regions(region_data, np.array(new_clipped_tif_files))
            
            # Get stats for region
            temporal_mean_region, temporal_std_region = read_stats(region=region_code)
            inundation_temporal[f"percent_inundation_{region_code}"] = np.nansum(region_area, axis=(1, 2)) / (total_cells - np.sum(np.isnan(region_area[0])))
            scaled_region_temporal_data = (inundation_temporal[f'percent_inundation_{region_code}'] - temporal_mean_region) / temporal_std_region
            inundation_temporal_scaled[f'percent_inundation_{region_code}'] = scaled_region_temporal_data

        # Combine existing and new inundation data
        with h5py.File('data/historic/inundation.h5', 'a') as hdf:
            dset = hdf['inundation']
            old_dataset_length = dset.shape[0]
            dset.resize(dset.shape[0] + len(new_clipped_tif_files), axis=0)
            dset[-len(new_clipped_tif_files):] = new_clipped_tif_files
            logging.info(f"Updated inundation data shape: {dset.shape}")
            
        # Update temporal data
        inundation_temporal_historic = pd.read_csv(temporal_data_path)[:old_dataset_length] # Crop to length of spatial data
        inundation_temporal_historic_scaled = pd.read_csv(temporal_data_path_scaled)[:old_dataset_length] # Crop to length of spatial data
        inundation_temporal_new = pd.concat([inundation_temporal_historic, inundation_temporal])
        inundation_temporal_new_scaled = pd.concat([inundation_temporal_historic_scaled, inundation_temporal_scaled])
        
        # Save the updated temporal data
        inundation_temporal_new['date'] = pd.to_datetime(inundation_temporal_new['date'], format='%Y-%m-%d')
        inundation_temporal_new.sort_values("date").to_csv('data/historic/inundation_temporal_unscaled.csv', index=False)
        inundation_temporal_new_scaled['date'] = pd.to_datetime(inundation_temporal_new_scaled['date'], format='%Y-%m-%d')
        inundation_temporal_new_scaled.sort_values("date").to_csv('data/historic/inundation_temporal_scaled.csv', index=False)
            
    except Exception as e:
        logging.error(f"Error processing new inundation data: {e}")

    logging.info("Inundation processing complete.\n")