# Import system libraries
import os
import glob

# Import cleaning utils
from .. import cleaning_utils
from ..config import get_cfg

# Import statistics
from data.stats import gridded_stats

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

DOWNLOADS_ROOT = get_cfg("paths.downloads.root", "data/downloads")
GR_RAINFALL_DOWNLOAD_PATH = get_cfg("paths.downloads.tamsat_rfe_daily", "data/downloads/tamsat/rfe/data/v3.1/daily")
EXTRACTED_DOMAIN_PATH = get_cfg("paths.downloads.extracted_domain", "data/downloads/extracted_data/domain")
GR_RAINFALL_DEKADS_PATH = get_cfg("paths.downloads.tamsat_rfe_dekads", "data/downloads/tamsat/rfe/dekads")
<<<<<<< HEAD
GR_RAINFALL_TEMPORAL_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall_temporal.csv")
GR_RAINFALL_H5_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall.h5")
=======
GR_RAINFALL_TEMPORAL_PATH = get_cfg("paths.historic.gridded_rainfall_temporal", "data/historic/gridded_rainfall_temporal.csv")
GR_RAINFALL_H5_PATH = get_cfg("paths.historic.gridded_rainfall_h5", "data/historic/gridded_rainfall.h5")
>>>>>>> origin/main
SAMPLE_TIF_FOLDER = get_cfg("paths.downloads.inundation_modis", "data/downloads/inundation_masks_modis")
CATCHMENTS_PATH = get_cfg("paths.maps.catchments", "data/maps/inflow_catchments/INFLOW_all_cmts.shp")


def reproject_rainfall(rainfall_ds, target_crs, target_transform, target_width, target_height):
    """
    Repoject and resample rainfall data.

    Parameters:
        rainfall_ds (array): NetCDF file with gridded rainfall data.
        target_crs (str): Target CRS for reprojection.
        target_transform (str): Target transformation for reprojection.
        target_width (float): Target width for reprojection.
        target_height (float): Target height for reprojection.
    """
    rainfall_data = rainfall_ds.read(1)  # Read the first band (rainfall data)
    rainfall_transform = rainfall_ds.transform
    rainfall_crs = rainfall_ds.crs

    # Reproject the rainfall data to match the target CRS, dimensions, and resolution
    reprojected_rainfall = np.empty((target_height, target_width), dtype=rainfall_data.dtype)
    reproject(
        source=rainfall_data,
        destination=reprojected_rainfall,
        src_transform=rainfall_transform,
        src_crs=rainfall_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear  # Bilinear resampling for continuous data (adjust if needed)
    )
    return reprojected_rainfall


def extract_date_from_filename(filename):
    """
    Extract the date from rainfall tif filename,

    Parameters:
        filname (str): Name of file for which the date is extracted.
    """
    return cleaning_utils.extract_date_from_tif_filename(filename)


def download_gridded_rainfall(dates_list, download_path):
    """
    Download gridded rainfall data for the specified dates.

    Parameters:
        dates_list (list): List of dates for which to download gridded rainfall data.
        download_path (str): Directory path to save downloaded NetCDF files.
    """
    try:
        download({
            "product": 'rfe',
            "timestep": 'daily',
            "resolution": 0.0375,
            "start_date": dates_list[0],
            "end_date": dates_list[-1],
            "version": 3.1,
            "localdata_dir": download_path
            })
    except Exception as e:
        print(f"Error occurred while downloading TAMSAT data: {e}")


def extract_gridded_rainfall(dates_list, download_path):
    """
    Extract gridded rainfall data for the specified dates into single file.

    Parameters:
        dates_list (list): List of dates for which to extract gridded rainfall data.
        download_path (str): Directory path to the downloaded NetCDF files.
    """
    try:
        extract({
            "product": 'rfe',
            "extract_type": 'domain',
            "N": 15.837321509670957,
            "S": -4.029166662242848,
            "W": 23.424907051000087,
            "E": 36.30367723700005,
            "timestep": 'daily',
            "resolution": 0.0375,
            "start_date": dates_list[0],
            "end_date": dates_list[-1],
            "version": 3.1,
            "localdata_dir": download_path
            })
    except Exception as e:
        raise RuntimeError(f"Failed to extract gridded rainfall data: {e}") from e


def _validate_rainfall_netcdf_file(netcdf_path):
    """Validate a domain rainfall NetCDF file and raise if unreadable/corrupt."""
    required_variables = {"lat", "lon", "rfe", "time"}
    try:
        with nc.Dataset(netcdf_path, mode='r') as ds:
            missing = required_variables - set(ds.variables.keys())
            if missing:
                raise RuntimeError(f"Missing required NetCDF variables: {sorted(missing)}")

            # Force a lightweight read to surface latent HDF issues early.
            _ = ds.variables['lat'][:1]
            _ = ds.variables['lon'][:1]
            _ = ds.variables['time'][:1]
            _ = ds.variables['rfe'][0:1, :, :]
    except Exception as e:
        raise RuntimeError(f"Invalid rainfall NetCDF file '{netcdf_path}': {e}") from e


def _get_latest_valid_rainfall_netcdf(extract_folder):
    """Return the most recent readable rainfall NetCDF from extract folder."""
    list_of_files = sorted(
        glob.glob(os.path.join(os.getcwd(), extract_folder, '*.nc')),
        key=os.path.getctime,
        reverse=True,
    )
    if not list_of_files:
        raise RuntimeError(f"No extracted NetCDF files found in '{extract_folder}'.")

    errors = []
    for path in list_of_files:
        try:
            _validate_rainfall_netcdf_file(path)
            return path
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {e}")

    raise RuntimeError(
        "No valid extracted rainfall NetCDF file found. Validation errors: "
        + " | ".join(errors)
    )


def get_historic_dates(data_path=GR_RAINFALL_TEMPORAL_PATH):
    """
    Get list of historic dates from pre-downloaded data.

    Parameters:
        data_path (str): Directory path of pre-downloaded temporal data.
    """
    try:
        gridded_rainfall_temporal = pd.read_csv(data_path)
        if "date" in gridded_rainfall_temporal.columns:
            historic_dates = gridded_rainfall_temporal["date"].astype(str).tolist()
        else:
            historic_dates = gridded_rainfall_temporal.index.astype(str).tolist()
        return historic_dates
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        return []


def download_new_gridded_rainfall(download_folder, target_product=None):
    """
    Download gridded rainfall data for dates not already downloaded.

    Parameters:
        download_folder (str): Directory folder to save downloaded TIF files.
    """
    target_product = cleaning_utils.resolve_target_product(target_product)
    download_path_full = os.path.join(os.getcwd(), download_folder)
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    historic_dates = get_historic_dates()
    has_historic_temporal = bool(historic_dates)

    if has_historic_temporal:
        last_date = historic_dates[-1]  # Get the last downloaded date
        last_date = datetime.strptime(last_date, "%Y-%m-%d").strftime("%Y-%m-%d")  # Ensure the format is YYYY-MM-DD
    else:
        # Bootstrap from the selected inundation target product timeline.
        last_date = cleaning_utils.get_target_start_date(target_product=target_product)

    new_dates = cleaning_utils.get_dates_of_interest(
        start_date_str=last_date,
        end_date_str=current_date_str,
        target_product=target_product,
    )

<<<<<<< HEAD
    local_product_path = os.path.join(os.getcwd(), GR_RAINFALL_DOWNLOAD_PATH)
    local_dates = cleaning_utils.get_local_download_dates(local_product_path)
    missing_dates = [d for d in new_dates if d not in local_dates]
=======
    if has_historic_temporal:
        local_dates = cleaning_utils.get_local_download_dates(download_path_full)
        missing_dates = [d for d in new_dates if d not in local_dates]
    else:
        # When no temporal record exists yet, rebuild the full historic range
        # instead of treating cached daily files as a complete processed record.
        missing_dates = new_dates
>>>>>>> origin/main

    if missing_dates:
        download_range = [missing_dates[0], missing_dates[-1]]
        download_gridded_rainfall(download_range, download_path_full)
        extract_gridded_rainfall(download_range, download_folder)
        # Fail-fast if extraction produced an unreadable or incomplete file.
        _get_latest_valid_rainfall_netcdf(EXTRACTED_DOMAIN_PATH)
    else:
        try:
            _get_latest_valid_rainfall_netcdf(EXTRACTED_DOMAIN_PATH)
        except RuntimeError:
            if local_dates:
                extract_gridded_rainfall([min(local_dates), max(local_dates)], download_folder)
                _get_latest_valid_rainfall_netcdf(EXTRACTED_DOMAIN_PATH)
            else:
                raise


def group_dates_by_decade(dates, target_product='modis'):
    """
    Group dates into 10-day intervals (dekads).

    Parameters:
        dates (pd.DatetimeIndex): Array of datetime objects.

    Returns:
        tuple: Grouped dates and their indices.
    """
    return cleaning_utils.group_dates_by_target_period(dates, target_product=target_product)


def export_decadal_geotiffs(extract_folder, output_folder, target_product='modis'):
    """
    Export rainfall data grouped by dekads into GeoTIFF files.

    Parameters:
        extract_folder (str): Path to folder where extracted rainfall data is saved.
        output_folder (str): Path to the folder to save GeoTIFFs.
    """
    # Ensure output directory exists before deleting/writing files.
    os.makedirs(output_folder, exist_ok=True)

    # Use glob to get all file paths in the folder
    files = glob.glob(os.path.join(output_folder, '*'))
    
    # Loop through the files and delete each one
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")
            
    # Select and validate latest extracted gridded rainfall file.
    latest_file = _get_latest_valid_rainfall_netcdf(extract_folder)

    # Stream from NetCDF in small slices so the full time cube is never loaded.
    with nc.Dataset(latest_file, mode='r') as rainfall_grid:
        lats = rainfall_grid.variables['lat'][:]
        lons = rainfall_grid.variables['lon'][:]
        times = rainfall_grid.variables['time'][:]
        rainfall_var = rainfall_grid.variables['rfe']

        first_date = datetime.strptime(latest_file[-24:-14], '%Y-%m-%d')
        dates = [(first_date + timedelta(days=int(i))) for i in times]

        print('--- Gridded rainfall data loaded ---')

        # Group the dates into target-product windows.
        date_groups, grouped_indices = group_dates_by_decade(dates, target_product=target_product)

        # Define transform once using the latitude and longitude arrays.
        lon_min = lons.min()
        lat_max = lats.max()
        pixel_size_x = lons[1] - lons[0]
        pixel_size_y = lats[1] - lats[0]
        transform = from_origin(lon_min, lat_max, pixel_size_x, -pixel_size_y)

        # Export each target-period group as a GeoTIFF.
        for group, indices in tqdm(zip(date_groups, grouped_indices), total=len(date_groups), desc="Exporting decadal averages"):
            indices = list(indices)
            if not indices:
                continue

            # Incremental mean across slices for low-memory processing.
            sum_2d = None
            for idx in indices:
                arr_2d = np.array(rainfall_var[idx, :, :], dtype=np.float32, copy=False)
                if sum_2d is None:
                    sum_2d = np.zeros_like(arr_2d, dtype=np.float32)
                sum_2d += arr_2d

            decadal_avg = sum_2d / float(len(indices))

            first_date = group[0]
            first_dekad_str = first_date.strftime("%Y%m%d")
            output_file = os.path.join(output_folder, f'rainfall_decadal_{first_dekad_str}.tif')

            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=decadal_avg.shape[0],
                width=decadal_avg.shape[1],
                count=1,
                dtype=decadal_avg.dtype,
                crs='EPSG:4326',
                transform=transform,
            ) as dst:
                dst.write(decadal_avg, 1)

<<<<<<< HEAD
            period_label = 'VIIRS half-month' if target_product == 'viirs' else 'MODIS dekad'
            print(f'Exported {period_label} GeoTIFF for {first_dekad_str}')
=======
            print(f'Exported decadal GeoTIFF for {first_dekad_str}')
>>>>>>> origin/main
        

def crop_historic_data(file_path, temporal_data_path):
    """
    Crop or recreate the historic inundation HDF5 dataset to match the temporal CSV length.

    If the HDF5 dataset is longer than the CSV, the entire HDF5 file will be truncated and recreated.
    """

    # --- Load temporal data length only ---
    hist = pd.read_csv(temporal_data_path)
    new_len = len(hist)

    # --- Open HDF5 and check dataset length ---
    with h5py.File(file_path, "r+") as f:
        dset_name = list(f.keys())[0]
        dset = f[dset_name]
        current_len = dset.shape[0]

        if current_len < new_len:
            hist.iloc[:current_len].to_csv(temporal_data_path, index=False)

        elif current_len > new_len:
            print(f"Cropping from {current_len} → {new_len} timesteps...")

            # 🔥 NEW: overwrite file entirely if HDF5 is longer than CSV
            f.close()  # Close open handle
            os.remove(file_path)  # Truncate file (delete completely)

            # Recreate the HDF5 file with the cropped data
            with h5py.File(file_path, "w") as newf:
                newf.create_dataset(
                    dset_name,
                    data=dset[:new_len],
                    maxshape=(None, *dset.shape[1:]),
                    chunks=True,
                    dtype=dset.dtype,
                )
            print("✅ File truncated and recreated with cropped data.")
        else:
            print("✅ No cropping needed. Temporal lengths already match.")


def process_new_gridded_rainfall(rainfall_dekads_folder,
                                 sample_tif_folder=SAMPLE_TIF_FOLDER,
                                 catchments_path=CATCHMENTS_PATH):
    """
    Process newly downloaded gridded rainfall data.

    Parameters:
        sample_tif_folder (str): Folder with sample inundation tif file for extracting boundaries.
        rainfall_dekads_folder (str): Folder with extracted rainfall dekads.
    """
    # List rainfall files and filter only the valid .tif files
    rainfall_dekads_files = [f for f in os.listdir(rainfall_dekads_folder) if f.endswith('.tif') and not f.endswith('(1).tif')]

    if not rainfall_dekads_files:
        logging.info("No grouped rainfall GeoTIFFs available to process yet.")
        return np.empty((0, *cleaning_utils.MASK_REGIONS_REF_SHAPE)), []

    # Sort the list of valid tif files based on their extracted date
    rainfall_files_sorted = sorted(rainfall_dekads_files, key=lambda f: extract_date_from_filename(f))
    rainfall_dekads_files_new = glob.glob(os.path.join(rainfall_dekads_folder, '*'))
    
    # Extract valid dates
    rainfall_dates = [extract_date_from_filename(f) for f in rainfall_dekads_files_new]
    rainfall_dates = [d for d in rainfall_dates if d is not None]

    # Create a DataFrame for alignment
    dates = pd.to_datetime([date.strftime('%Y-%m-%d') for date in rainfall_dates])
    dates_df = pd.DataFrame({'date': dates}).sort_values('date').reset_index()
    sorted_dates = list(dates_df['date'])
    rainfall_df = pd.DataFrame({'rainfall_file': rainfall_dekads_files_new, 'rainfall_date': rainfall_dates}).sort_values('rainfall_date').reset_index()

    # Merge the two dataframes to ensure every target date has a corresponding rainfall date
    aligned_df = pd.merge(dates_df, rainfall_df, left_on='date', right_on='rainfall_date', how='left').sort_values('rainfall_date').reset_index()

    # Check for missing dates
    missing_rainfall_dates = aligned_df[aligned_df['rainfall_file'].isna()]['date']
    if not missing_rainfall_dates.empty:
        print(f"Warning: Missing rainfall data for {len(missing_rainfall_dates)} dates.")

    # Align rainfall files
    aligned_rainfall_files = aligned_df['rainfall_file'].tolist()

    return aligned_rainfall_files, sorted_dates


def process_single_rainfall_tif(rainfall_tif_path, catchments):
    """Align and mask a single rainfall GeoTIFF onto the shared reference grid."""
    with rasterio.open(rainfall_tif_path) as rainfall_ds:
        resampled_rainfall = cleaning_utils.align_and_mask_raster_to_reference_grid(
            src=rainfall_ds,
            mask_gdf=catchments,
            src_band=1,
            dst_fill=0,
            resampling=Resampling.bilinear,
        )
    if resampled_rainfall is None:
        raise RuntimeError(f"Reprojection returned None for file: {rainfall_tif_path}")
    return resampled_rainfall.astype(np.float32, copy=False)


def update_gridded_rainfall(
    download_folder=DOWNLOADS_ROOT,
    download_path=GR_RAINFALL_DOWNLOAD_PATH,
    extract_folder=EXTRACTED_DOMAIN_PATH,
    dekads_path=GR_RAINFALL_DEKADS_PATH,
    temporal_data_path=GR_RAINFALL_TEMPORAL_PATH):
    """
    Combine newly downloaded gridded rainfall with existing data.

    Parameters:
        download_folder (str): Directory folder to save downloaded TIF files.
        download_path (str): Directory path to save downloaded TIF files.
        extract_folder (str): Directory folder to extracted TIF files.
        dekads_path (str): Directory path to export dekadal TIF files.
        temporal_data_path (str): Directory path to historic temporal data CSV.
    """
    try:
        target_product = cleaning_utils.resolve_target_product(None)

        # Crop historic data if historic spatial and temporal data are not the same size   
        if os.path.exists(GR_RAINFALL_H5_PATH) and os.path.exists(temporal_data_path):
            crop_historic_data(
                file_path=GR_RAINFALL_H5_PATH,
                temporal_data_path=temporal_data_path,
            )
        
        # Update rainfall data
        download_new_gridded_rainfall(download_folder, target_product=target_product)
        
        # Process new files
        dekads_path_full = os.path.join(os.getcwd(), dekads_path)
        export_decadal_geotiffs(extract_folder, dekads_path_full, target_product=target_product)
        aligned_files, dates = process_new_gridded_rainfall(dekads_path_full)
        historic_dates = get_historic_dates()

        # Identify new files
        pending = []
        for i in range(len(dates)):
            date_str = dates[i].strftime("%Y-%m-%d")
            tif_name = aligned_files[i]
            if date_str in historic_dates:
                continue
            if pd.isna(tif_name):
                logging.warning(f"Skipping missing rainfall file for date {date_str}")
                continue
            pending.append((os.path.join(dekads_path_full, tif_name), date_str))

        if len(pending) == 0:
            logging.info("No new files to process.")

        else:
            # Prepare region masks and output state.
            regions_gdf = cleaning_utils.extract_regions()
            catchments = gpd.read_file(CATCHMENTS_PATH)

            if os.path.exists(GR_RAINFALL_H5_PATH):
                hdf_mode = 'a'
            else:
                hdf_mode = 'w'

            temporal_rows = []
            with h5py.File(GR_RAINFALL_H5_PATH, hdf_mode) as hdf:
                dset = hdf.get('rainfall')

                for tif_path, date_str in tqdm(pending, desc="Streaming rainfall into H5"):
                    try:
                        rainfall_2d = process_single_rainfall_tif(tif_path, catchments)
                    except Exception as e:
                        raise RuntimeError(f"Failed processing rainfall tif '{tif_path}': {e}") from e

                    if dset is None:
                        dset = hdf.create_dataset(
                            'rainfall',
                            shape=(0, rainfall_2d.shape[0], rainfall_2d.shape[1]),
                            maxshape=(None, rainfall_2d.shape[0], rainfall_2d.shape[1]),
                            chunks=(1, rainfall_2d.shape[0], rainfall_2d.shape[1]),
                            dtype=np.float32,
                        )

                    current_len = dset.shape[0]
                    dset.resize(current_len + 1, axis=0)
                    dset[current_len] = rainfall_2d

                    total_cells = rainfall_2d.shape[0] * rainfall_2d.shape[1]
                    row = {
                        'date': date_str,
                        'rainfall': float(np.nansum(rainfall_2d)),
                    }
                    rainfall_3d = rainfall_2d[np.newaxis, :, :]
                    for i in range(len(regions_gdf)):
                        region_data = regions_gdf.iloc[[i]]
                        region_code = gridded_stats.region_to_code_dict[region_data['region'].values[0]]
                        region_area = cleaning_utils.mask_regions(region_data, rainfall_3d)
                        valid_cells = total_cells - np.sum(np.isnan(region_area[0]))
                        row[f"rainfall_{region_code}"] = (
                            float(np.nansum(region_area)) / valid_cells if valid_cells > 0 else np.nan
                        )

                    temporal_rows.append(row)

                logging.info(f"Updated rainfall dataset shape: {dset.shape}")

            # Update temporal data
            rainfall_temporal = pd.DataFrame(temporal_rows)
            if os.path.exists(temporal_data_path):
                rainfall_temporal_historic = pd.read_csv(temporal_data_path)
            else:
                rainfall_temporal_historic = pd.DataFrame(columns=rainfall_temporal.columns)
            rainfall_temporal_new = pd.concat([rainfall_temporal_historic, rainfall_temporal], ignore_index=True)

            # Save the updated temporal data
            rainfall_temporal_new['date'] = pd.to_datetime(rainfall_temporal_new['date'], errors='coerce')
            rainfall_temporal_new = rainfall_temporal_new.dropna(subset=['date'])
            rainfall_temporal_new = rainfall_temporal_new.sort_values('date').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
            rainfall_temporal_new.to_csv(temporal_data_path, index=False)

    except Exception as e:
        logging.error(f"Error processing rainfall data: {e}")
        raise