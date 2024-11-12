import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Define file paths
FOLDER_PATH = "inundation_masks_updated"
CATCHMENTS_PATH = "Project Map and Shapefiles/INFLOW_cmts_clean/INFLOW_cmts_15/INFLOW_all_cmts.shp"
ELEVATION_TIF_PATH = "dem/study_area_dem.tif"
OUTPUT_PATH = "outputs/"

# Step 1: Load catchments and ensure CRS matches a sample raster file
def load_catchments_and_align_crs(catchments_path, folder_path):
    catchments = gpd.read_file(catchments_path)
    sample_tif_path = os.path.join(folder_path, [f for f in os.listdir(folder_path) if f.endswith(".tif")][0])
    with rasterio.open(sample_tif_path) as src:
        catchments = catchments.to_crs(src.crs)
    return catchments, sample_tif_path

# Step 2: Clip a sample raster to get spatial attributes (transform, dimensions, bounds, CRS)
def get_sample_raster_attributes(sample_tif_path, catchments):
    with rasterio.open(sample_tif_path) as src:
        clipped, clipped_transform = rasterio_mask(src, catchments.geometry, crop=True)
        sample_height, sample_width = clipped.shape[1], clipped.shape[2]
        sample_crs = src.crs
        sample_bounds = rasterio.transform.array_bounds(sample_height, sample_width, clipped_transform)
    return sample_width, sample_height, sample_crs, sample_bounds, clipped_transform

# Step 3: Load and reproject elevation data to match sample CRS if necessary
def reproject_elevation(elevation_tif_path, target_crs):
    with rasterio.open(elevation_tif_path) as elevation_ds:
        elevation_data = elevation_ds.read(1)
        elevation_transform = elevation_ds.transform
        elevation_crs = elevation_ds.crs

        if elevation_crs != target_crs:
            transform, width, height = calculate_default_transform(
                elevation_crs, target_crs, elevation_ds.width, elevation_ds.height, *elevation_ds.bounds
            )
            reprojected_elevation = np.empty((height, width), dtype=elevation_data.dtype)
            reproject(
                source=elevation_data,
                destination=reprojected_elevation,
                src_transform=elevation_transform,
                src_crs=elevation_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
            return reprojected_elevation, transform, target_crs
        else:
            return elevation_data, elevation_transform, elevation_crs

# Step 4: Resample elevation data to match sample raster's spatial resolution and bounds
def resample_elevation_to_sample_bounds(elevation_data, elevation_transform, elevation_crs, sample_bounds, sample_crs, sample_width, sample_height):
    transform, width, height = calculate_default_transform(
        elevation_crs, sample_crs, sample_width, sample_height, *sample_bounds
    )
    resampled_elevation = np.empty((height, width), dtype=elevation_data.dtype)
    reproject(
        source=elevation_data,
        destination=resampled_elevation,
        src_transform=elevation_transform,
        src_crs=elevation_crs,
        dst_transform=transform,
        dst_crs=sample_crs,
        resampling=Resampling.nearest
    )
    return resampled_elevation, transform

# Step 5: Standardize elevation data
def standardize_data(data):
    return (data - np.mean(data)) / np.std(data)

# Step 6: Save the elevation array with an added dimension for consistency
def save_elevation_array(elevation_data, output_path):
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'elevation.npy'), elevation_data)

# Main processing function
def process_elevation():
    # Load and align catchments CRS with sample raster
    catchments, sample_tif_path = load_catchments_and_align_crs(CATCHMENTS_PATH, FOLDER_PATH)
    
    # Get sample raster attributes
    sample_width, sample_height, sample_crs, sample_bounds, sample_transform = get_sample_raster_attributes(sample_tif_path, catchments)
    
    # Reproject elevation data to sample CRS if needed
    elevation_data, elevation_transform, elevation_crs = reproject_elevation(ELEVATION_TIF_PATH, sample_crs)
    
    # Resample elevation data to match the sample raster's bounds and resolution
    resampled_elevation, resampled_transform = resample_elevation_to_sample_bounds(
        elevation_data, elevation_transform, elevation_crs, sample_bounds, sample_crs, sample_width, sample_height
    )
    
    # Add an extra dimension for consistency with other datasets
    elevation = np.expand_dims(resampled_elevation, axis=0)
    
    # Standardize elevation data
    elevation = standardize_data(elevation)
    
    # Save the standardized elevation array
    save_elevation_array(elevation, OUTPUT_PATH)
    print("Elevation processing and resampling completed.")