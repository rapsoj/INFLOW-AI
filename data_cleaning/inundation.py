import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from datetime import datetime
from tqdm import tqdm
import requests

def download_inundation(dates_list, download_path='inundation_masks_updated'):
    """
    Download inundation data for the specified dates.
    """
    base_url = "https://data.earthobservation.vam.wfp.org/public-share/sudd_dashboard/ssdmask/ssdmask"
    os.makedirs(download_path, exist_ok=True)

    for date in tqdm(dates_list, desc="Downloading inundation data"):
        date = datetime.strptime(date, '%Y-%m-%d') if isinstance(date, str) else date
        formatted_date = date.strftime('%Y%m%d')
        file_url = f"{base_url}{formatted_date}.tif"
        file_path = os.path.join(download_path, f"{formatted_date}.tif")

        try:
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Error occurred while downloading {formatted_date}.tif: {e}")

def get_sorted_tif_files(folder_path):
    """
    Get a sorted list of TIF files in a specified folder.
    """
    tif_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".tif"))
    return tif_files

def load_shapefile(path):
    """
    Load a shapefile as a GeoDataFrame.
    """
    return gpd.read_file(path)

def reproject_to_raster_crs(shapefile, raster_path):
    """
    Reproject a GeoDataFrame to match the CRS of a raster file.
    """
    with rasterio.open(raster_path) as src:
        return shapefile.to_crs(src.crs)

def process_and_clip_rasters(tif_files, folder_path, catchments):
    """
    Clip and collect metadata for each raster file in a folder.
    """
    clipped_tif_files, tif_file_names, spatial_metadata = [], [], {}

    for file_name in tqdm(tif_files, desc="Processing TIF files"):
        file_path = os.path.join(folder_path, file_name)

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
    return clipped_tif_files, tif_file_names, spatial_metadata

def save_inundation_array(clipped_tif_files, output_path='outputs/inundation.npy'):
    """
    Save a list of clipped raster arrays as a single numpy array.
    """
    inundation = np.array(clipped_tif_files)
    np.save(output_path, inundation)
    return inundation

def initialize_and_reproject_shapefile(catchments_path, folder_path):
    """
    Load and reproject the shapefile to match the CRS of the first TIF file.
    """
    catchments = load_shapefile(catchments_path)
    first_raster_path = os.path.join(folder_path, get_sorted_tif_files(folder_path)[0])
    return reproject_to_raster_crs(catchments, first_raster_path)

def process_inundation():
    """
    Main function to process inundation data from TIF files.
    """
    folder_path = "inundation_masks_updated"
    catchments_path = "Project Map and Shapefiles/INFLOW_cmts_clean/INFLOW_cmts_15/INFLOW_all_cmts.shp"

    catchments = initialize_and_reproject_shapefile(catchments_path, folder_path)
    tif_files = get_sorted_tif_files(folder_path)
    clipped_tif_files, tif_file_names, spatial_metadata = process_and_clip_rasters(tif_files, folder_path, catchments)

    inundation = save_inundation_array(clipped_tif_files)
    print("Processing and saving completed.")
    print("Inundation array shape:", inundation.shape)
    return inundation

def update_inundation(dates_list, download_path='inundation_masks_updated'):
    """
    Download inundation data for dates not already downloaded.
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    sorted_files = get_sorted_tif_files(download_path)

    if not sorted_files:
        print("No TIF files found. Downloading all dates in the list.")
        new_dates = dates_list
    else:
        last_file = sorted_files[-1]
        last_date = datetime.strptime(last_file[:8], "%Y%m%d")
        new_dates = [date for date in dates_list if date > last_date]

    download_inundation(new_dates, download_path)

def process_new_inundation(dates_list, download_path='inundation_masks_updated'):
    """
    Process newly downloaded inundation data and combine it with existing data.
    """
    existing_inundation = np.load('outputs/inundation.npy')
    existing_dates = {datetime.strptime(f[:8], "%Y%m%d") for f in get_sorted_tif_files(download_path)}

    update_inundation(dates_list, download_path)

    sorted_files = get_sorted_tif_files(download_path)
    new_files = [f for f in sorted_files if datetime.strptime(f[:8], "%Y%m%d") not in existing_dates]

    if not new_files:
        print("No new files to process.")
        return existing_inundation

    catchments_path = "Project Map and Shapefiles/INFLOW_cmts_clean/INFLOW_cmts_15/INFLOW_all_cmts.shp"
    catchments = initialize_and_reproject_shapefile(catchments_path, download_path)
    new_clipped_tif_files, _, _ = process_and_clip_rasters(new_files, download_path, catchments)

    combined_inundation = np.concatenate((existing_inundation, np.array(new_clipped_tif_files)), axis=0)
    np.save('outputs/inundation.npy', combined_inundation)
    print("Updated inundation data shape:", combined_inundation.shape)

    return combined_inundation
