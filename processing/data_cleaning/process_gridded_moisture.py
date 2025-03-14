# Import system libraries
import os
import glob

# Import cleaning utils
from .. import cleaning_utils

# Import statistics
from data.stats import gridded_data_stats

# Import TAMSAT API
from processing.data_cleaning.download_tamsat.tamsat_download_extract_api import download, extract

# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import geospatial libraries
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import netCDF4 as nc
import xarray as xr

# Import client libraries
import wget

# Import compression libraries
import h5py

# Import progress bar libraries
from tqdm import tqdm

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_stats(region='all'):
    """
    Read the gridded data statistics file.
    """
    moisture_mean = gridded_data_stats.gridded_moisture_stats[region]['mean']
    moisture_std = gridded_data_stats.gridded_moisture_stats[region]['std']
    
    return moisture_mean, moisture_std


def standardize_array(array, mean, std):
    """
    Standard scale the 3D array.

    Parameters:
        array (array): Array to be standard scaled.
        mean (float): Mean value for standardisation.
        std (float): Standard deviation value for standardisation.
    """
    # Apply standardization
    standardized_array = (array - mean) / std
    return standardized_array


def reproject_moisture(moisture_ds, target_crs, target_transform, target_width, target_height):
    """
    Repoject and resample moisture data.

    Parameters:
        moisture_ds (array): NetCDF file with gridded moisture data.
        target_crs (str): Target CRS for reprojection.
        target_transform (str): Target transformation for reprojection.
        target_width (float): Target width for reprojection.
        target_height (float): Target height for reprojection.
    """
    moisture_data = moisture_ds.read(1)  # Read the first band (moisture data)
    moisture_transform = moisture_ds.transform
    moisture_crs = moisture_ds.crs

    # Reproject the moisture data to match the target CRS, dimensions, and resolution
    reprojected_moisture = np.empty((target_height, target_width), dtype=moisture_data.dtype)
    reproject(
        source=moisture_data,
        destination=reprojected_moisture,
        src_transform=moisture_transform,
        src_crs=moisture_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear  # Bilinear resampling for continuous data (adjust if needed)
    )
    return reprojected_moisture


def extract_date_from_filename(filename):
    """
    Extract the date from moisture tif filename,

    Parameters:
        filname (str): Name of file for which the date is extracted.
    """
    if filename.endswith('.tif'):
        # Split by '_' and handle cases where extra characters (like ' (1)') are added
        date_str = filename.split('_')[-1].split('.')[0]  # Extract the 'YYYYMMDD' part and ignore anything after '.'
        # Remove any non-digit characters (in case of extra numbering like "(1)")
        date_str = ''.join(filter(str.isdigit, date_str))
        try:
            return pd.to_datetime(date_str, format='%Y%m%d')
        except ValueError:
            return None  # Return None if date extraction fails
    else:
        return None  # Skip non-tif files


def download_gridded_moisture(dates_list, download_path):
    """
    Download gridded moisture data for the specified dates.

    Parameters:
        dates_list (list): List of dates for which to download gridded moisture data.
        download_path (str): Directory path to save downloaded NetCDF files.
    """
    try:
        download({
            "product": 'sm',
            "timestep": 'daily',
            "resolution": 0.25,
            "start_date": dates_list[0],
            "end_date": dates_list[-1],
            "version": '2.3.1',
            "localdata_dir": download_path
            })
    except Exception as e:
        print(f"Error occurred while downloading TAMSAT data: {e}")


def extract_gridded_moisture(dates_list, download_path):
    """
    Extract gridded moisture data for the specified dates into single file.

    Parameters:
        dates_list (list): List of dates for which to extract gridded moisture data.
        download_path (str): Directory path to the downloaded NetCDF files.
    """
    try:
        extract({
            "product": 'sm',
            "extract_type": 'domain',
            "N": 15.837321509670957,
            "S": -4.029166662242848,
            "W": 23.424907051000087,
            "E": 36.30367723700005,
            "timestep": 'daily',
            "resolution": 0.25,
            "start_date": dates_list[0],
            "end_date": dates_list[-1],
            "version": '2.3.1',
            "localdata_dir": download_path
            })

    except Exception as e:
        print(f"Error occurred while extracting gridded moisture data: {e}")


def get_historic_dates(data_path='data/historic/gridded_moisture_temporal.csv'):
    """
    Get list of historic dates from pre-downloaded data.

    Parameters:
        data_path (str): Directory path of pre-downloaded temporal data.
    """
    try:
        gridded_moisture_temporal = pd.read_csv(data_path, index_col=0)
        historic_dates = gridded_moisture_temporal.index.tolist()
        return historic_dates
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        return []


def download_new_gridded_moisture(download_folder):
    """
    Download gridded moisture data for dates not already downloaded.

    Parameters:
        download_folder (str): Directory folder to save downloaded TIF files.
    """
    download_path_full = os.path.join(os.getcwd(), download_folder)
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    historic_dates = get_historic_dates()

    if historic_dates:
        last_date = historic_dates[-1]  # Get the last downloaded date
        last_date = datetime.strptime(last_date, "%Y-%m-%d").strftime("%Y-%m-%d")  # Ensure the format is YYYY-MM-DD
    else:
        last_date = datetime.now().strftime("%Y-%m-%d")

    new_dates = cleaning_utils.get_dates_of_interest(start_date_str=last_date, end_date_str=current_date_str)

    if new_dates:
        download_gridded_moisture(new_dates, download_path_full)
        extract_gridded_moisture(new_dates, download_folder)
    else:
        logging.info("No new dates to download.")


def group_dates_by_decade(dates):
    """
    Group dates into 10-day intervals (dekads).

    Parameters:
        dates (pd.DatetimeIndex): Array of datetime objects.

    Returns:
        tuple: Grouped dates and their indices.
    """
    date_groups = []
    grouped_indices = []
    current_group = []
    current_indices = []

    for i, date in enumerate(dates):
        day = date.day

        if day == 1 or day == 11 or day == 21:
            if current_group:
                date_groups.append(current_group)
                grouped_indices.append(current_indices)
            current_group = [date]
            current_indices = [i]
        else:
            current_group.append(date)
            current_indices.append(i)

    if current_group:
        date_groups.append(current_group)
        grouped_indices.append(current_indices)

    # Remove incomplete dekads
    grouped_indices = [group for group in grouped_indices if len(group) >= 8]
    
    return date_groups, grouped_indices


def export_decadal_geotiffs(extract_folder, output_folder):
    """
    Export moisture data grouped by dekads into GeoTIFF files.

    Parameters:
        extract_folder (str): Path to folder where extracted moisture data is saved.
        output_folder (str): Path to the folder to save GeoTIFFs.
    """
    # Use glob to get all file paths in the folder
    files = glob.glob(os.path.join(output_folder, '*'))
    
    # Loop through the files and delete each one
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")
            
    # Get latest extracted gridded moisture file
    list_of_files = glob.glob(os.path.join(os.getcwd(), extract_folder, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)

    # Open the extracted gridded moisture file
    moisture_grid = nc.Dataset(latest_file, mode='r')

    # Extract latitude, longitude, and moisture data
    lats = moisture_grid.variables['lat'][:]
    lons = moisture_grid.variables['lon'][:]
    moisture = moisture_grid.variables['sm_c4grass'][:]
    times = moisture_grid.variables['time'][:]
    # Extract dates
    first_date = datetime.strptime(latest_file[-24:-14], '%Y-%m-%d')
    dates = [(first_date + timedelta(days=int(i))) for i in times]

    print('--- Gridded moisture data loaded ---')

    # Close the NetCDF file
    moisture_grid.close()

    # Define resolution
    resolution_x = lons[1] - lons[0]
    resolution_y = lats[1] - lats[0]

    # Calculate the spatial extent
    min_lon, max_lon = lons.min(), lons.max()
    max_lat, min_lat = lats.max(), lats.min()
    
    # Group the dates into decades (1-10, 11-20, 21-end)
    date_groups, grouped_indices = group_dates_by_decade(dates)

    # Export each decadal group as a GeoTIFF
    for group, indices in tqdm(zip(date_groups, grouped_indices), total=len(date_groups), desc="Exporting decadal averages"):
        # Calculate the average moisture for the current group of dates
        decadal_avg = np.mean(moisture[indices, :, :], axis=0)

        # Use the first date in the group for the file naming
        first_date = group[0]
        first_dekad_str = first_date.strftime("%Y%m%d")

        # Define output file path for each decadal period
        output_file = os.path.join(output_folder, f'moisture_decadal_{first_dekad_str}.tif')

        # Define transform using the latitude and longitude arrays
        lon_min = lons.min()
        lat_max = lats.max()
        lat_min = lats.min()
        pixel_size_x = lons[1] - lons[0]
        pixel_size_y = lats[1] - lats[0]  # Use positive pixel size for Y

        # Create transform
        transform = from_origin(lon_min, lat_min, pixel_size_x, -pixel_size_y)

        # Open a new GeoTIFF file
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=decadal_avg.shape[0],
            width=decadal_avg.shape[1],
            count=1,  # Single band (moisture data)
            dtype=decadal_avg.dtype,
            crs='EPSG:4326',  # Assuming lat/lon coordinates
            transform=transform,
        ) as dst:
            # Write the averaged data for the decadal period
            dst.write(decadal_avg, 1)

        print(f'Exported decadal GeoTIFF for {first_dekad_str}')


def process_new_gridded_moisture(moisture_dekads_folder,
                                 sample_tif_folder='data/downloads/inundation_masks',
                                 catchments_path="data/maps/inflow_catchments/INFLOW_all_cmts.shp"):
    """
    Process newly downloaded gridded moisture data.

    Parameters:
        sample_tif_folder (str): Folder with sample inundation tif file for extracting boundaries.
        moisture_dekads_folder (str): Folder with extracted moisture dekads.
    """
    # List moisture files and filter only the valid .tif files
    moisture_dekads_files = [f for f in os.listdir(moisture_dekads_folder) if f.endswith('.tif') and not f.endswith('(1).tif')]

    # Sort the list of valid tif files based on their extracted date
    moisture_files_sorted = sorted(moisture_dekads_files, key=lambda f: extract_date_from_filename(f))
    moisture_dekads_files_new = glob.glob(os.path.join(moisture_dekads_folder, '*'))
    
    # Extract valid dates
    moisture_dates = [extract_date_from_filename(f) for f in moisture_dekads_files_new]
    moisture_dates = [d for d in moisture_dates if d is not None]

    # Create a DataFrame for alignment
    dates = pd.to_datetime([date.strftime('%Y-%m-%d') for date in moisture_dates])
    dates_df = pd.DataFrame({'date': dates}).sort_values('date').reset_index()
    sorted_dates = list(dates_df['date'])
    moisture_df = pd.DataFrame({'moisture_file': moisture_dekads_files_new, 'moisture_date': moisture_dates}).sort_values('moisture_date').reset_index()

    # Merge the two dataframes to ensure every MODIS date has a corresponding moisture date
    aligned_df = pd.merge(dates_df, moisture_df, left_on='date', right_on='moisture_date', how='left').sort_values('moisture_date').reset_index()

    # Check for missing dates
    missing_moisture_dates = aligned_df[aligned_df['moisture_file'].isna()]['date']
    if not missing_moisture_dates.empty:
        print(f"Warning: Missing moisture data for {len(missing_moisture_dates)} dates.")

    # Align moisture files
    aligned_moisture_files = aligned_df['moisture_file'].tolist()

    # List to store processed decadal moisture data
    moisture_data_list = []

    # Read catchments shapefile
    catchments = gpd.read_file(catchments_path)

    # Ensure catchments CRS matches the sample tif
    def ensure_crs_match(geodf, raster_file):
        with rasterio.open(raster_file) as src:
            return geodf.to_crs(src.crs)
    
    sample_tif_path = os.path.join(sample_tif_folder, os.listdir(sample_tif_folder)[0])
    catchments = ensure_crs_match(catchments, sample_tif_path)

    # Read the sample tif file to get its transform and dimensions
    with rasterio.open(sample_tif_path) as src:
        clipped, clipped_transform = rasterio_mask(src, catchments.geometry, crop=True)
        sample_width = clipped.shape[2]
        sample_height = clipped.shape[1]
        sample_crs = src.crs
        sample_bounds = rasterio.transform.array_bounds(sample_height, sample_width, clipped_transform)
        sample_res = (src.res[0], src.res[1])  # Get the resolution of the sample tif (pixel size)

    # Proceed with processing aligned moisture files
    for moisture_tif in tqdm(aligned_moisture_files, desc="Processing aligned moisture TIF files"):

        try:
            if pd.notna(moisture_tif):  # Only process valid tif files
                moisture_tif_path = os.path.join(moisture_dekads_folder, moisture_tif)
                
                # Open moisture GeoTIFF
                with rasterio.open(moisture_tif_path) as moisture_ds:
                    # Reproject and resample moisture data to match the sample tif's CRS, bounds, and resolution
                    resampled_moisture = reproject_moisture(
                        moisture_ds,
                        target_crs=sample_crs,
                        target_transform=clipped_transform,
                        target_width=sample_width,
                        target_height=sample_height
                    )
    
                # Convert resampled moisture data to array
                if resampled_moisture is not None:
                    moisture_data_list.append(resampled_moisture)
                else:
                    logging.error(f"Reprojection returned None for file: {moisture_tif_path}")
            else:
                logging.error(f"Skipping invalid or NaN entry: {moisture_tif}")
    
        except FileNotFoundError as fnf_error:
            logging.error(f"File not found: {moisture_tif_path}. Error: {fnf_error}")
            continue  # Proceed to next iteration if file not found
    
        except rasterio.errors.RasterioError as raster_error:
            logging.error(f"Error opening or processing GeoTIFF file: {moisture_tif_path}. Error: {raster_error}")
            continue  # Proceed to next iteration if error opening the file
    
        except Exception as e:
            logging.error(f"Unexpected error processing file {moisture_tif_path}: {e}")
            continue  # Proceed to next iteration for any unexpected error

    # Convert the list to a 3D array (time, lat, lon)
    gridded_moisture_new = np.stack(moisture_data_list, axis=0)

    # Standard scale new data based on saved values
    moisture_mean, moisture_std = read_stats()
    gridded_moisture_new = standardize_array(gridded_moisture_new, moisture_mean, moisture_std)
    
    return gridded_moisture_new, sorted_dates


def update_gridded_moisture(
        download_folder='data/downloads', 
        download_path='data/downloads/tamsat/soil_moisture/data/v2.3.1/daily',
        extract_folder='data/downloads/extracted_data/domain',
        dekads_path='data/downloads/tamsat/soil_moisture/dekads',
        temporal_data_path='data/historic/gridded_moisture_temporal.csv'):
    """
    Combine newly downloaded gridded moisture with existing data.

    Parameters:
        download_folder (str): Directory folder to save downloaded TIF files.
        download_path (str): Directory path to save downloaded TIF files.
        extract_folder (str): Directory folder to extracted TIF files.
        dekads_path (str): Directory path to export dekadal TIF files.
        temporal_data_path (str): Directory path to historic temporal data CSV.
    """
    try:
        # Update moisture data
        download_new_gridded_moisture(download_folder)
        
        # Process new files
        dekads_path_full = os.path.join(os.getcwd(), dekads_path)
        export_decadal_geotiffs(extract_folder, dekads_path_full)
        sorted_files, dates = process_new_gridded_moisture(dekads_path_full)
        historic_dates = get_historic_dates()

        # Identify new files
        new_data = np.array([sorted_files[i] for i in range(len(dates)) if dates[i].strftime("%Y-%m-%d") not in historic_dates])
        new_dates = [ts.strftime("%Y-%m-%d") for ts in dates if ts.strftime("%Y-%m-%d") not in historic_dates]
        
        if len(new_data) == 0:
            logging.info("No new files to process.")
            
        else:
            # Crop area to regions of interest
            regions_gdf = cleaning_utils.extract_regions()
            
            # Calculate total number of cells
            total_cells = new_data[0].shape[0] * new_data[0].shape[1]
            
            # Create new temporal data
            moisture_temporal = pd.DataFrame({'moisture': new_data.sum(axis=(1, 2))})
            moisture_temporal['date'] = new_dates
            temporal_mean, temporal_std = read_stats(region='all_temporal')
            moisture_temporal['moisture'] = (moisture_temporal['moisture'] - temporal_mean) / temporal_std
            
            # Loop through regions
            for i in range(len(regions_gdf)):
                region_data = regions_gdf.iloc[[i]]
                region_code = gridded_data_stats.region_to_code_dict[region_data['region'].values[0]]
                region_area = cleaning_utils.mask_regions(region_data, np.array(new_data))
                
                # Get stats for region
                temporal_mean_region, temporal_std_region = read_stats(region=region_code)
                moisture_temporal[f"moisture_{region_code}"] = np.nansum(region_area, axis=(1, 2)) / (total_cells - np.sum(np.isnan(region_area[0])))
                moisture_temporal[f"moisture_{region_code}"] = (moisture_temporal[f"moisture_{region_code}"] - temporal_mean_region) / temporal_std_region
                
            # Update temporal data
            moisture_temporal_historic = pd.read_csv(temporal_data_path)
            moisture_temporal_new = pd.concat([moisture_temporal_historic, moisture_temporal])
            
            # Save the updated temporal data
            moisture_temporal_new.to_csv(temporal_data_path, index=False)
    
            # Append new data to HDF5
            with h5py.File('data/historic/gridded_moisture.h5', 'a') as hdf:
                dset = hdf['moisture']
                dset.resize(dset.shape[0] + new_data.shape[0], axis=0)
                dset[-new_data.shape[0]:] = new_data
                logging.info(f"Updated moisture dataset shape: {dset.shape}")

    except Exception as e:
        logging.error(f"Error processing moisture data: {e}")