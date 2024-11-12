import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import Polygon, MultiPolygon, box
from scipy.ndimage import label

# Paths and constants (can be passed as parameters if needed)
shapefile_path = 'Project Map and Shapefiles/INFLOW_cmts_clean/INFLOW_cmts_15/INFLOW_all_cmts.shp'
folder_path = "inundation_masks_updated"
catchment_ids = [11, 12, 13, 14, 15]

def clean_geometry(geom, buffer_value=1e-9):
    """Fix invalid geometries and remove holes."""
    geom = geom.buffer(0)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda polygon: polygon.area)
    if isinstance(geom, Polygon):
        geom = Polygon(geom.exterior)
    return geom.buffer(buffer_value)

def load_and_clean_catchments():
    """Load and clean catchments based on predefined catchment IDs."""
    catchments = gpd.read_file(shapefile_path)
    for id in catchment_ids:
        catchment_geom = catchments.loc[catchments['INFLOW_ID'] == id, 'geometry'].iloc[0]
        catchments.loc[catchments['INFLOW_ID'] == id, 'geometry'] = clean_geometry(catchment_geom)
    return catchments

def reproject_catchments(catchments):
    """Reproject catchments to match the CRS of a sample raster in the folder."""
    sample_tif = next(f for f in os.listdir(folder_path) if f.endswith(".tif"))
    file_path = os.path.join(folder_path, sample_tif)
    with rasterio.open(file_path) as src:
        catchments = catchments.to_crs(src.crs)
    return catchments, file_path

def process_raster(catchments, file_path):
    """Clip and mask raster data for each catchment and return combined array."""
    catchment_id_data = []
    with rasterio.open(file_path) as src:
        raster_crs = src.crs
        for id in catchment_ids:
            catchments_study = catchments[catchments['INFLOW_ID'] == id].reset_index()
            if catchments_study.crs != raster_crs:
                catchments_study = catchments_study.to_crs(raster_crs)
            src_clipped, src_clipped_transform = rasterio_mask(src, catchments.geometry, crop=True)
            mask = features.geometry_mask(
                catchments_study.geometry,
                transform=src_clipped_transform,
                invert=True,
                out_shape=(src_clipped.shape[1], src_clipped.shape[2])
            )
            raster_data = src_clipped[0]
            raster_data[mask] = 1
            raster_data[~mask] = 0
            catchment_id_data.append(raster_data)
    return np.array(catchment_id_data)

def identify_holes(catchment_array):
    """Identify and fill holes in the catchment array."""
    data = catchment_array.sum(axis=0)
    inverted_grid = 1 - data
    labeled_grid, num_features = label(inverted_grid)

    hole_coords_flat = [
        (row, col)
        for i in range(1, num_features + 1)
        if not is_on_boundary(np.where(labeled_grid == i), data.shape)
        for row, col in zip(*np.where(labeled_grid == i))
    ]
    for row, col in hole_coords_flat:
        catchment_array[0, row, col] = 1

def is_on_boundary(component, grid_shape):
    """Check if a component is touching the boundary of the grid."""
    rows, cols = grid_shape
    return (np.any(component[0] == 0) or np.any(component[0] == rows - 1) or
            np.any(component[1] == 0) or np.any(component[1] == cols - 1))

def process_catchments():
    """Main function to process catchments and return the final catchment array."""
    # Load and clean catchments
    catchments = load_and_clean_catchments()

    # Reproject catchments and get file path of the reference raster
    catchments, file_path = reproject_catchments(catchments)

    # Process raster data for catchment IDs
    catchment_id = process_raster(catchments, file_path)

    # Identify and fill holes in the data
    identify_holes(catchment_id)

    # Save the final catchment array
    np.save('outputs/catchment_array.npy', catchment_id)
    print("Catchment array saved as 'outputs/catchment_array.npy'.")
    return catchment_id
