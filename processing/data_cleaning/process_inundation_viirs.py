# Import system libraries
import os
import re
import calendar
import logging
import time
from datetime import datetime
from urllib.parse import urljoin

# Import data manipulation libraries
import numpy as np
import pandas as pd

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
from ..config import get_cfg

# Import statistics
from data.stats import gridded_data_stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
# VIIRS configuration
# -----------------------------------------------------------------------------

VIIRS_BASE_URL = get_cfg(
    "sources.viirs_base_url",
    "https://edcftp.cr.usgs.gov/project/FEWSNET/"
    "spervez/Togofiles/SouthSudan/UofReading/Geotiffs/",
)

VIIRS_PERIOD_ORDER = {
    "Mid": 0,
    "Shm": 1,
    "End": 2,
}

VIIRS_DOWNLOAD_PATH = get_cfg("paths.downloads.inundation_viirs", "data/downloads/inundation_masks_viirs")
VIIRS_H5_PATH = get_cfg("paths.historic.viirs_h5", "data/historic/viirs_inundation.h5")
VIIRS_TEMPORAL_UNSCALED_PATH = get_cfg(
    "paths.historic.viirs_temporal_unscaled",
    "data/historic/viirs_inundation_temporal_unscaled.csv",
)
VIIRS_TEMPORAL_SCALED_PATH = get_cfg(
    "paths.historic.viirs_temporal_scaled",
    "data/historic/viirs_inundation_temporal_scaled.csv",
)
VIIRS_DSET_NAME = "inundation"
INFLOW_CATCHMENTS_PATH = get_cfg(
    "paths.maps.catchments",
    "data/maps/inflow_catchments/INFLOW_all_cmts.shp",
)


def read_stats(region='all'):
    """
    Read the gridded data statistics file.
    """
    inundation_mean = gridded_data_stats.inundation_stats[region]['mean']
    inundation_std = gridded_data_stats.inundation_stats[region]['std']

    return inundation_mean, inundation_std


def parse_viirs_filename(file_name):
    """
    Parse a VIIRS inundation filename.

    Expected patterns:
        FldYYYYMid_MM.tif
        FldYYYYShm_MM.tif
        FldYYYYEnd_MM.tif

    Returns:
        dict with year, month, period_type, period_order, period_id, and date fields.
    """
    fld_pattern = r"^Fld(?P<year>\d{4})(?P<period>Mid|Shm|End)_(?P<month>\d{2})\.tif$"
    date_pattern = r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.tif$"

    fld_match = re.match(fld_pattern, file_name, flags=re.IGNORECASE)
    date_match = re.match(date_pattern, file_name, flags=re.IGNORECASE)

    if fld_match:
        year = int(fld_match.group("year"))
        month = int(fld_match.group("month"))
        period_type = fld_match.group("period").capitalize()
        period_order = VIIRS_PERIOD_ORDER[period_type]
        _, last_day = calendar.monthrange(year, month)

        if period_type == "Mid":
            period_start = datetime(year, month, 1).date()
            period_end = datetime(year, month, 15).date()
        elif period_type == "Shm":
            period_start = datetime(year, month, 16).date()
            period_end = datetime(year, month, last_day).date()
        else:  # End
            period_start = datetime(year, month, 1).date()
            period_end = datetime(year, month, last_day).date()

        period_id = f"{year:04d}-{month:02d}-{period_type.lower()}"
    elif date_match:
        year = int(date_match.group("year"))
        month = int(date_match.group("month"))
        day = int(date_match.group("day"))
        _, last_day = calendar.monthrange(year, month)
        period_start = datetime(year, month, day).date()

        if day == 1:
            period_type = "Mid"
            period_order = VIIRS_PERIOD_ORDER[period_type]
            period_end = datetime(year, month, 15).date()
            period_id = f"{year:04d}-{month:02d}-mid"
        elif day == 16:
            period_type = "Shm"
            period_order = VIIRS_PERIOD_ORDER[period_type]
            period_end = datetime(year, month, last_day).date()
            period_id = f"{year:04d}-{month:02d}-shm"
        else:
            period_type = "Unknown"
            period_order = 99
            period_end = period_start
            period_id = period_start.isoformat()
    else:
        raise ValueError(f"Unrecognized VIIRS filename format: {file_name}")

    return {
        "year": year,
        "month": month,
        "period_type": period_type,
        "period_order": period_order,
        "period_id": period_id,
        "period_start": period_start,
        "period_end": period_end,
    }


def viirs_target_filename(file_name):
    """
    Convert a remote VIIRS file name to the local target filename.

    Local convention uses period start date:
      FldYYYYMid_MM.tif -> YYYY-MM-01.tif
      FldYYYYShm_MM.tif -> YYYY-MM-16.tif

    End-of-month files are ignored and return None.
    """
    parsed = parse_viirs_filename(file_name)
    if parsed["period_type"] == "End":
        return None
    return f"{parsed['period_start'].isoformat()}.tif"


def normalize_viirs_identifier(identifier):
    """
    Normalize a VIIRS identifier to local filename convention when possible.
    """
    if not isinstance(identifier, str):
        return identifier

    if not identifier.lower().endswith(".tif"):
        return identifier

    try:
        target = viirs_target_filename(identifier)
        return target if target is not None else identifier
    except ValueError:
        return identifier


def sort_viirs_files(file_names):
    """
    Sort VIIRS files chronologically by year, month, and period type.
    """
    def _sort_key(file_name):
        parsed = parse_viirs_filename(file_name)
        return (parsed["period_start"], parsed["period_order"], file_name)

    return sorted(file_names, key=_sort_key)


def list_remote_tif_files(base_url=VIIRS_BASE_URL):
    """
    List remote VIIRS TIF files in the source directory.
    """
    try:
        response = requests.get(base_url, timeout=60)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Error listing remote VIIRS files from {base_url}: {e}")
        return []

    # Try to parse direct links from the HTML directory listing.
    tif_files = re.findall(r'href=["\']?([^"\'>]+\.tif)["\']?', response.text, flags=re.IGNORECASE)
    tif_files = [os.path.basename(f) for f in tif_files if f.lower().endswith(".tif")]

    # Fallback to simple filename scraping if needed.
    if not tif_files:
        tif_files = re.findall(r'(Fld\d{4}(?:Mid|Shm|End)_\d{2}\.tif)', response.text, flags=re.IGNORECASE)

    tif_files = list(dict.fromkeys(os.path.basename(f) for f in tif_files))

    # Keep only expected VIIRS files and ignore End files.
    filtered_files = []
    for file_name in tif_files:
        try:
            parsed = parse_viirs_filename(file_name)
            if parsed["period_type"] != "End":
                filtered_files.append(file_name)
        except ValueError:
            continue

    return sort_viirs_files(filtered_files)


def download_inundation(file_list, download_path=VIIRS_DOWNLOAD_PATH, base_url=VIIRS_BASE_URL):
    """
    Download VIIRS inundation data for the specified files.

    Parameters:
        file_list (list): List of VIIRS TIF file names to download.
        download_path (str): Directory path to save downloaded TIF files.
        base_url (str): Base URL where the TIF files are hosted.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    max_retries = 3
    retry_delay_seconds = 2

    for file_name in tqdm(file_list, desc="Downloading VIIRS inundation data"):
        file_url = urljoin(base_url, file_name)
        target_name = viirs_target_filename(file_name)
        if target_name is None:
            continue
        file_path = os.path.join(download_path, target_name)
        temp_file_path = f"{file_path}.part"

        if os.path.exists(file_path):
            continue

        for attempt in range(1, max_retries + 1):
            try:
                with requests.get(file_url, stream=True, timeout=120) as response:
                    if response.status_code != 200:
                        raise requests.HTTPError(
                            f"HTTP {response.status_code} for {file_name}",
                            response=response,
                        )

                    with open(temp_file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                os.replace(temp_file_path, file_path)
                break

            except Exception as e:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                if attempt == max_retries:
                    logging.error(
                        f"Failed to download {file_name} after {max_retries} attempts: {e}"
                    )
                else:
                    logging.warning(
                        f"Retry {attempt}/{max_retries} for {file_name} after error: {e}"
                    )
                    time.sleep(retry_delay_seconds)


def get_sorted_tif_files(folder_path):
    """
    Get a sorted list of TIF files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing TIF files.

    Returns:
        list: Sorted list of TIF file names.
    """
    try:
        tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tif")]
        return sort_viirs_files(tif_files)
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


def get_historic_dates(data_path=VIIRS_TEMPORAL_UNSCALED_PATH):
    """
    Get list of historic VIIRS records from pre-downloaded data.

    Returns the most useful unique identifier available in the temporal CSV:
    file_name, period_id, or date.
    """
    try:
        inundation_temporal = pd.read_csv(data_path)

        if "file_name" in inundation_temporal.columns:
            return inundation_temporal["file_name"].tolist()
        if "period_id" in inundation_temporal.columns:
            return inundation_temporal["period_id"].tolist()
        if "date" in inundation_temporal.columns:
            return inundation_temporal["date"].astype(str).tolist()

        # Fallback: use the first column or index values if present.
        if inundation_temporal.shape[1] > 0:
            return inundation_temporal.iloc[:, 0].astype(str).tolist()

        return []
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        return []


def download_new_inundation(download_path=VIIRS_DOWNLOAD_PATH, burn_in_steps=18):
    """
    Download VIIRS inundation data for the last `burn_in_steps` files (to refresh them)
    plus any new files available in the remote directory.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        burn_in_steps (int): Number of timesteps to always refresh.
    """
    remote_files = list_remote_tif_files()
    if not remote_files:
        logging.info("No remote VIIRS files found.")
        return

    remote_items = [(f, viirs_target_filename(f)) for f in remote_files]
    remote_items = [(src, tgt) for src, tgt in remote_items if tgt is not None]

    historic_ids = {normalize_viirs_identifier(x) for x in get_historic_dates()}
    if historic_ids:
        if len(remote_items) >= burn_in_steps:
            refresh_files = [src for src, _ in remote_items[-burn_in_steps:]]
        else:
            refresh_files = [src for src, _ in remote_items]
    else:
        refresh_files = []

    new_files = [src for src, tgt in remote_items if tgt not in historic_ids]
    files_to_download = []
    seen = set()
    for file_name in refresh_files + new_files:
        if file_name not in seen:
            files_to_download.append(file_name)
            seen.add(file_name)

    if files_to_download:
        logging.info(
            f"Downloading {len(files_to_download)} VIIRS files "
            f"(including last {burn_in_steps} for refresh)."
        )
        download_inundation(files_to_download, download_path)
    else:
        logging.info("No new VIIRS files to download.")


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


def remove_burn_in_data(h5_file_path=VIIRS_H5_PATH,
                        temporal_data_path=VIIRS_TEMPORAL_UNSCALED_PATH,
                        temporal_data_path_scaled=VIIRS_TEMPORAL_SCALED_PATH,
                        dset_name=VIIRS_DSET_NAME,
                        burn_in_steps=18):
    """
    Remove the last `burn_in_steps` timesteps from saved VIIRS data
    (spatio-temporal HDF5 dataset and temporal CSVs).

    Parameters:
        h5_file_path (str): Path to spatio-temporal historic VIIRS HDF5 file.
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

        # Sort chronologically if VIIRS period metadata exists
        sort_cols = [c for c in ["year", "month", "period_order"] if c in df_cropped.columns]
        if sort_cols:
            df_cropped = df_cropped.sort_values(sort_cols).reset_index(drop=True)
        elif "date" in df_cropped.columns:
            df_cropped["date"] = pd.to_datetime(df_cropped["date"])
            df_cropped = df_cropped.sort_values("date").reset_index(drop=True)

        # Save back to CSV
        df_cropped.to_csv(csv_path, index=False)

    print(f"Removed last {burn_in_steps} timesteps from HDF5 and temporal CSVs.")


def build_viirs_temporal_dataframe(file_names, clipped_rasters, regions_gdf=None):
    """
    Build unscaled and scaled temporal dataframes from VIIRS rasters.
    """
    if not clipped_rasters:
        raise ValueError("No clipped VIIRS rasters were provided.")

    parsed_records = [parse_viirs_filename(file_name) for file_name in file_names]
    total_cells = clipped_rasters[0].shape[0] * clipped_rasters[0].shape[1]

    inundation_temporal = pd.DataFrame({
        "file_name": file_names,
        "period_id": [rec["period_id"] for rec in parsed_records],
        "year": [rec["year"] for rec in parsed_records],
        "month": [rec["month"] for rec in parsed_records],
        "period_type": [rec["period_type"] for rec in parsed_records],
        "period_order": [rec["period_order"] for rec in parsed_records],
        "period_start": [rec["period_start"] for rec in parsed_records],
        "period_end": [rec["period_end"] for rec in parsed_records],
        "percent_inundation": np.sum(clipped_rasters, axis=(1, 2)) / total_cells,
    })

    inundation_temporal_scaled = inundation_temporal.copy()
    temporal_mean, temporal_std = read_stats()
    inundation_temporal_scaled["percent_inundation"] = (
        inundation_temporal_scaled["percent_inundation"] - temporal_mean
    ) / temporal_std

    if regions_gdf is not None and len(regions_gdf) > 0:
        for i in range(len(regions_gdf)):
            region_data = regions_gdf.iloc[[i]]
            region_code = gridded_data_stats.region_to_code_dict[region_data["region"].values[0]]
            region_area = cleaning_utils.mask_regions(region_data, np.array(clipped_rasters))

            temporal_mean_region, temporal_std_region = read_stats(region=region_code)
            region_series = np.nansum(region_area, axis=(1, 2)) / (
                total_cells - np.sum(np.isnan(region_area[0]))
            )
            inundation_temporal[f"percent_inundation_{region_code}"] = region_series
            inundation_temporal_scaled[f"percent_inundation_{region_code}"] = (
                region_series - temporal_mean_region
            ) / temporal_std_region

    sort_cols = [c for c in ["year", "month", "period_order"] if c in inundation_temporal.columns]
    if sort_cols:
        inundation_temporal = inundation_temporal.sort_values(sort_cols).reset_index(drop=True)
        inundation_temporal_scaled = inundation_temporal_scaled.sort_values(sort_cols).reset_index(drop=True)

    return inundation_temporal, inundation_temporal_scaled


def update_inundation(h5_file_path=VIIRS_H5_PATH,
                      download_path=VIIRS_DOWNLOAD_PATH,
                      temporal_data_path=VIIRS_TEMPORAL_UNSCALED_PATH,
                      temporal_data_path_scaled=VIIRS_TEMPORAL_SCALED_PATH):
    """
    Process newly downloaded VIIRS data and combine it with existing data.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        temporal_data_path (str): Directory path of pre-downloaded temporal data.
        temporal_data_path_scaled (str): Directory path of pre-downloaded scaled temporal data.
    """
    try:
        has_existing_h5 = False
        if os.path.exists(h5_file_path):
            with h5py.File(h5_file_path, 'r') as f:
                if VIIRS_DSET_NAME in f:
                    viirs_historic = f[VIIRS_DSET_NAME]
                    logging.info(f"Existing VIIRS data shape: {viirs_historic.shape}")
                    has_existing_h5 = True

        if has_existing_h5:
            # Crop historic data if historic spatial and temporal data are not the same size
            crop_historic_data(
                file_path=h5_file_path,
                temporal_data_path=temporal_data_path,
                temporal_data_path_scaled=temporal_data_path_scaled
            )

            # Remove burn-in data
            remove_burn_in_data()
        else:
            logging.info(
                f"Historic VIIRS dataset not found at {h5_file_path}. "
                "Bootstrapping dataset from newly processed files."
            )

        # Update VIIRS data by downloading new files
        download_new_inundation(download_path)

        # Get the sorted TIF files after the update
        sorted_files = get_sorted_tif_files(download_path)
        historic_ids = {normalize_viirs_identifier(x) for x in get_historic_dates()}

        # Identify new files to process (files not already processed)
        new_files = [file_name for file_name in sorted_files if file_name not in historic_ids]

        if not new_files:
            logging.info("No new files to process.")
            return

        # Process the new TIF files
        catchments = load_shapefile(INFLOW_CATCHMENTS_PATH)
        first_raster_path = os.path.join(download_path, sorted_files[0])
        catchments = reproject_to_raster_crs(catchments, first_raster_path)

        # Process rasters and gather new data
        new_clipped_tif_files, new_file_names, _ = process_and_clip_rasters(new_files, download_path, catchments)

        # Crop area to regions of interest
        regions_gdf = cleaning_utils.extract_regions()

        # Build temporal dataframes
        viirs_temporal, viirs_temporal_scaled = build_viirs_temporal_dataframe(
            new_file_names,
            new_clipped_tif_files,
            regions_gdf=regions_gdf
        )

        # Combine existing and new VIIRS data
        os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)
        with h5py.File(h5_file_path, 'a') as hdf:
            if VIIRS_DSET_NAME in hdf:
                dset = hdf[VIIRS_DSET_NAME]
                old_dataset_length = dset.shape[0]
                dset.resize(dset.shape[0] + len(new_clipped_tif_files), axis=0)
                dset[-len(new_clipped_tif_files):] = new_clipped_tif_files
            else:
                old_dataset_length = 0
                dset = hdf.create_dataset(
                    VIIRS_DSET_NAME,
                    data=np.array(new_clipped_tif_files),
                    maxshape=(None, *np.array(new_clipped_tif_files).shape[1:]),
                    chunks=True,
                )
            logging.info(f"Updated VIIRS data shape: {dset.shape}")

        # Update temporal data
        if old_dataset_length > 0 and os.path.exists(temporal_data_path):
            viirs_temporal_historic = pd.read_csv(temporal_data_path)[:old_dataset_length]
        else:
            viirs_temporal_historic = pd.DataFrame(columns=viirs_temporal.columns)

        if old_dataset_length > 0 and os.path.exists(temporal_data_path_scaled):
            viirs_temporal_historic_scaled = pd.read_csv(temporal_data_path_scaled)[:old_dataset_length]
        else:
            viirs_temporal_historic_scaled = pd.DataFrame(columns=viirs_temporal_scaled.columns)
        viirs_temporal_new = pd.concat([viirs_temporal_historic, viirs_temporal], ignore_index=True)
        viirs_temporal_new_scaled = pd.concat([viirs_temporal_historic_scaled, viirs_temporal_scaled], ignore_index=True)

        # Save the updated temporal data
        sort_cols = [c for c in ["year", "month", "period_order"] if c in viirs_temporal_new.columns]
        if sort_cols:
            viirs_temporal_new = viirs_temporal_new.sort_values(sort_cols).reset_index(drop=True)
            viirs_temporal_new_scaled = viirs_temporal_new_scaled.sort_values(sort_cols).reset_index(drop=True)
        elif "date" in viirs_temporal_new.columns:
            viirs_temporal_new["date"] = pd.to_datetime(viirs_temporal_new["date"], errors="coerce")
            viirs_temporal_new = viirs_temporal_new.sort_values("date").reset_index(drop=True)
            viirs_temporal_new_scaled["date"] = pd.to_datetime(viirs_temporal_new_scaled["date"], errors="coerce")
            viirs_temporal_new_scaled = viirs_temporal_new_scaled.sort_values("date").reset_index(drop=True)

        os.makedirs(os.path.dirname(VIIRS_TEMPORAL_UNSCALED_PATH), exist_ok=True)
        viirs_temporal_new.to_csv(VIIRS_TEMPORAL_UNSCALED_PATH, index=False)
        viirs_temporal_new_scaled.to_csv(VIIRS_TEMPORAL_SCALED_PATH, index=False)

    except Exception as e:
        logging.error(f"Error processing new VIIRS data: {e}")

    logging.info("VIIRS inundation processing complete.\\n")