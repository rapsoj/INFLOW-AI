import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.features import rasterize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define file paths
FOLDER_PATH = "inundation_masks_updated"
CATCHMENTS_PATH = "Project Map and Shapefiles/INFLOW_cmts_clean/INFLOW_cmts_15/INFLOW_all_cmts.shp"
BASIN_PATH = 'hydroatlas/inflow-basin-atlas.gpkg'
OUTPUT_PATH = 'outputs/'

# Load and ensure CRS consistency
def load_and_align_crs(catchments_path, basins_path, raster_folder):
    catchments = gpd.read_file(catchments_path)
    basins = gpd.read_file(basins_path)
    sample_raster = os.path.join(raster_folder, [f for f in os.listdir(raster_folder) if f.endswith(".tif")][0])
    with rasterio.open(sample_raster) as src:
        catchments = catchments.to_crs(src.crs)
        basins = basins.to_crs(src.crs)
    return catchments, basins, sample_raster

# Process raster with scaling and masking
def process_raster(file_path, catchments, scale_factor=10):
    with rasterio.open(file_path) as src:
        clipped, clipped_transform = rasterio_mask(src, catchments.geometry, crop=True)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "driver": "GTiff",
            "height": clipped.shape[1] // scale_factor,
            "width": clipped.shape[2] // scale_factor,
            "transform": clipped_transform,
        })
        return clipped, clipped_meta, clipped_transform

# Rasterize basin polygons to match raster dimensions
def rasterize_basins(basins, transform, shape):
    shapes = [(geom, basin_id) for geom, basin_id in zip(basins.geometry, basins['HYBAS_ID'])]
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.int32)

# Filter and encode categorical columns in basins data
def prepare_basin_attributes(basins, polygon_grid, cols_of_interest):
    basins_filtered = basins[basins['HYBAS_ID'].isin(polygon_grid.flatten())][cols_of_interest]
    
    categorical_columns = [col for col in basins_filtered.columns if '_cl_' in col]
    basins_filtered[categorical_columns] = basins_filtered[categorical_columns].astype('category')
    
    one_hot_encoded = pd.get_dummies(basins_filtered[categorical_columns], drop_first=True)
    basins_filtered = pd.concat([basins_filtered.drop(columns=categorical_columns), one_hot_encoded], axis=1)
    
    return basins_filtered

# Scale non-categorical columns and apply PCA
def scale_and_reduce(basins_filtered):
    noncategorical_columns = [col for col in basins_filtered.columns if '_cl_' not in col]
    
    scaler = StandardScaler()
    scaled_noncat_data = scaler.fit_transform(basins_filtered[noncategorical_columns])
    scaled_noncat_df = pd.DataFrame(scaled_noncat_data, columns=noncategorical_columns, index=basins_filtered.index)
    
    basins_filtered_encoded = pd.get_dummies(basins_filtered[[col for col in basins_filtered.columns if col not in noncategorical_columns]], drop_first=True)
    df_combined = pd.concat([scaled_noncat_df, basins_filtered_encoded], axis=1)
    
    pca = PCA(n_components=min(df_combined.shape))
    principal_components = pca.fit_transform(df_combined)
    significant_components = [i for i, ratio in enumerate(pca.explained_variance_ratio_) if ratio >= 0.01]
    
    principal_components_df = pd.DataFrame(
        principal_components[:, significant_components],
        index=basins_filtered.index,
        columns=[f'basins_PC{i+1}' for i in significant_components]
    )
    
    return principal_components_df

# Create 3D basin attributes array from polygon grid and attributes
def create_3d_basin_array(basins_filtered, polygon_grid):
    id_to_value_maps = {col: basins_filtered[col].to_dict() for col in basins_filtered.columns}
    n, x, y = len(basins_filtered.columns), *polygon_grid.shape
    basin_attributes = np.zeros((n, x, y), dtype=np.float32)
    
    for i, column in enumerate(basins_filtered.columns):
        id_to_value_map = id_to_value_maps[column]
        for id_value, replacement in id_to_value_map.items():
            if isinstance(replacement, (int, float)):
                basin_attributes[i][polygon_grid == id_value] = replacement
            else:
                raise ValueError(f"Replacement value for ID {id_value} in column {column} is not numeric: {replacement}")
    return basin_attributes

# Save output arrays
def save_output(basin_attributes, polygon_grid):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    np.save(os.path.join(OUTPUT_PATH, 'basin_attributes.npy'), basin_attributes)
    np.save(os.path.join(OUTPUT_PATH, 'polygon_grid.npy'), polygon_grid)

# Main processing function
def process_basin_attributes():
    # Load and ensure CRS match
    catchments, basins, sample_raster = load_and_align_crs(CATCHMENTS_PATH, BASIN_PATH, FOLDER_PATH)
    
    # Process raster and identify parameters
    clipped, clipped_meta, clipped_transform = process_raster(sample_raster, catchments)
    polygon_grid = rasterize_basins(basins, clipped_transform, np.squeeze(clipped, axis=0).shape)

    # Define columns of interest and process basin attributes
    cols_of_interest = [
        "DIST_SINK", "DIST_MAIN", "glc_pc_u15", "wet_pc_sg1", "wet_pc_s04", "wet_pc_sg2",
        "wet_pc_ug1", "inu_pc_slt", "wet_pc_u04", "wet_pc_ug2", "inu_pc_smx", "inu_pc_ult",
        "soc_th_sav", "inu_pc_umx", "lka_pc_use", "dis_m3_pmn", "wet_pc_u01", "lkv_mc_usu",
        "glc_pc_u02", "tmp_dc_smn", "tmp_dc_s12", "gwt_cm_sav", "tmp_dc_s11", "pop_ct_usu",
        "inu_pc_smn", "lka_pc_sse", "soc_th_uav", "gdp_ud_usu", "dis_m3_pyr", "tmp_dc_s02",
        "tmp_dc_s01", "glc_pc_u20", "tmp_dc_s03", "ele_mt_sav", "ele_mt_smx", "ele_mt_smn",
        "inu_pc_umn", "riv_tc_usu", "tmp_dc_syr", "ria_ha_usu", "tmp_dc_s10", "slp_dg_sav",
        "tmp_dc_s08", "riv_tc_ssu", "ria_ha_ssu", "wet_pc_s03", "tmp_dc_s09", "pre_mm_s02",
        "wet_pc_s01", "tmp_dc_s04", "cmi_ix_s02", "glc_cl_smj_15", "tbi_cl_smj_9",
        "tec_cl_smj_74", "tbi_cl_smj_7", "wet_cl_smj_4", "lit_cl_smj_8", "clz_cl_smj_17",
        "clz_cl_smj_18", "fec_cl_smj_522", "cls_cl_smj_123", "fmh_cl_smj_10", "tec_cl_smj_43",
        "tec_cl_smj_52", "cls_cl_smj_122", "tec_cl_smj_53", "glc_cl_smj_3", "glc_cl_smj_9",
        "pnv_cl_smj_9"
    ]
    basins_filtered = prepare_basin_attributes(basins, polygon_grid, cols_of_interest)
    basins_filtered = scale_and_reduce(basins_filtered)
    
    # Create and save the 3D array
    basin_attributes = create_3d_basin_array(basins_filtered, polygon_grid)
    save_output(basin_attributes, polygon_grid)
    print("Basin attributes processing and saving completed.")