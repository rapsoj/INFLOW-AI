# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import geospatial libraries
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
import geopandas as gpd

# Import machine learning libraries
from sklearn.linear_model import LinearRegression

# Import config
from .config import get_cfg


# -----------------------------------------------------------------------------
# Static reference grid for region masking
# Derived from: data/downloads/inundation_masks_modis/20020801.tif
# with catchment crop applied using data/maps/inflow_catchments/INFLOW_all_cmts.shp
# -----------------------------------------------------------------------------
MASK_REGIONS_REF_CRS = get_cfg("reference_grid.mask_regions.crs", "EPSG:6933")
MASK_REGIONS_REF_TRANSFORM = tuple(
    get_cfg("reference_grid.mask_regions.transform", [1000.0, 0.0, 2260000.0, 0.0, -1000.0, 1558000.0])
)
MASK_REGIONS_REF_SHAPE = tuple(get_cfg("reference_grid.mask_regions.shape", [1125, 1204]))
MASK_REGIONS_REF_BOUNDS = tuple(
    get_cfg("reference_grid.mask_regions.bounds", [2260000.0, 433000.0, 3464000.0, 1558000.0])
)


def _reference_transform_affine():
    """Return the configured static reference transform as an Affine object."""
    return Affine(*MASK_REGIONS_REF_TRANSFORM)


def rasterize_to_reference_grid(gdf, all_touched=True, dtype=np.uint8):
    """
    Rasterize geometries directly on the static reference grid.

    Parameters:
    - gdf: GeoDataFrame
        Input geometries to rasterize.
    - all_touched: bool
        Whether all touched pixels are burned in.
    - dtype: numpy dtype
        Output dtype for the rasterized mask.

    Returns:
    - ndarray
        2D mask in reference grid where geometry pixels are 1 and other pixels are 0.
    """
    ref_crs = MASK_REGIONS_REF_CRS

    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)

    return rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=MASK_REGIONS_REF_SHAPE,
        transform=MASK_REGIONS_REF_TRANSFORM,
        fill=0,
        all_touched=all_touched,
        dtype=dtype,
    )


def align_raster_to_reference_grid(src, src_band=1, resampling=Resampling.nearest, dst_fill=0):
    """
    Reproject and resample any georeferenced raster band to the static reference grid.

    This aligns by CRS and writes directly into the configured reference extent,
    transform, and shape, ensuring all outputs share identical footprint and grid.

    Parameters:
    - src: rasterio.io.DatasetReader
        Open raster dataset.
    - src_band: int
        Source band index.
    - resampling: rasterio.warp.Resampling
        Resampling method for reprojection.
    - dst_fill: int or float
        Fill value for pixels outside source coverage.

    Returns:
    - ndarray
        2D aligned array in the static reference grid.
    """
    dst = np.full(MASK_REGIONS_REF_SHAPE, dst_fill, dtype=src.dtypes[src_band - 1])

    reproject(
        source=rasterio.band(src, src_band),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=_reference_transform_affine(),
        dst_crs=MASK_REGIONS_REF_CRS,
        src_nodata=src.nodata,
        dst_nodata=dst_fill,
        resampling=resampling,
    )

    return dst


def align_and_mask_raster_to_reference_grid(src, mask_gdf, src_band=1, dst_fill=0):
    """
    Align raster to reference grid and apply a polygon mask in that same grid.

    Parameters:
    - src: rasterio.io.DatasetReader
        Open raster dataset.
    - mask_gdf: GeoDataFrame
        Mask geometries (e.g., catchments).
    - src_band: int
        Source band index.
    - dst_fill: int or float
        Fill value outside source and mask coverage.

    Returns:
    - ndarray
        2D aligned+masked array in the static reference grid.
    """
    aligned = align_raster_to_reference_grid(src=src, src_band=src_band, dst_fill=dst_fill)
    mask = rasterize_to_reference_grid(mask_gdf, all_touched=True, dtype=np.uint8)
    return np.where(mask == 1, aligned, dst_fill)

ADMIN0_PATH = get_cfg(
    "paths.maps.admin0",
    "data/maps/admin_boundaries/ssd_admbnda_adm0_imwg_nbs_20230829.shp",
)
ADMIN1_PATH = get_cfg(
    "paths.maps.admin1",
    "data/maps/admin_boundaries/ssd_admbnda_adm1_imwg_nbs_20230829.shp",
)
ABYEI_PATH = get_cfg(
    "paths.maps.abyei",
    "data/maps/abyei_region/ssd_admbnda_abyei_imwg_nbs_20180401.shp",
)


def get_dates_of_interest(start_date_str='2002-07-01', end_date_str=None):
    """
    Generate a list of dates between start_date_str and end_date_str where the day ends in '01', '11', or '21'.

    Parameters:
        start_date_str (str): The start date in 'YYYY-MM-DD' format. Defaults to '2002-07-01'.
        end_date_str (str): The end date in 'YYYY-MM-DD' format. Defaults to 60 days from today if not provided.

    Returns:
        list: A list of dates (in 'YYYY-MM-DD' format) where the day is 1, 11, or 21.
    """
    # Parse the start date
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid start_date_str format. Expected 'YYYY-MM-DD', got: {start_date_str}")
    
    # Get today's date if end_date_str is not provided, otherwise parse the end_date_str
    if not end_date_str:
        end_date = datetime.today()
    else:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid end_date_str format. Expected 'YYYY-MM-DD', got: {end_date_str}")
    
    # Initialize an empty list to store the dates
    dates_of_interest = []

    # Iterate through all dates between start_date and end_date
    current_date = start_date
    while current_date <= end_date:
        # Check if the day ends in '01', '11', or '21'
        if current_date.day in [1, 11, 21]:
            dates_of_interest.append(current_date.strftime('%Y-%m-%d'))
        # Move to the next day
        current_date += timedelta(days=1)

    return dates_of_interest


# Define function to linearly extrapolate missing values between data points
def impute_missing_values(df, cols, regression_length=6):
    """
    Impute missing values for each column in the provided dataframe using linear regression.

    Parameters:
    - df: pd.DataFrame
        The dataframe containing time series data with missing values.
    - cols: list
        A list of column names to impute using linear regression.
    - regression_length: int, default=6
        The number of past non-missing data points to use for linear regression.

    Returns:
    - df_imputed: pd.DataFrame
        The dataframe with missing values imputed for the specified columns.
    """
    # Get the current date
    current_date = pd.Timestamp.now()

    # Filter dates that are before the current date
    original_index = df.index
    df.index = pd.to_datetime(df.index)
    dates = df.index
    past_dates = dates[dates <= current_date]
    data = df.loc[past_dates]

    # Loop through each column to impute missing values
    for col in cols:
        # Split the data for the current column
        past_data = data[data[col].notna()][-regression_length:]  # Last non-missing values
        impute_data = data[data[col].isna()]  # Data with NaNs to impute
        forecast_steps = len(impute_data)

        if forecast_steps == 0 or len(past_data) < regression_length:
            # If there are no missing values to impute or not enough past data, skip this column
            continue

        # Prepare data for linear regression
        print(f"Imputing {forecast_steps} timestep for {col}.")
        series = past_data[col]
        X = np.arange(len(series)).reshape(-1, 1)  # Time index (0, 1, ..., n)
        y = series.values  # Corresponding values

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Create future indices based on the missing data indices
        future_indices = impute_data.index  # Get indices for missing dates

        # Generate future time indices for predictions (match length of forecast steps)
        future_time_steps = np.arange(len(series), len(series) + forecast_steps).reshape(-1, 1)
        forecast_values = model.predict(future_time_steps)

        # Create a Series for the forecast values (imputed points)
        forecast_df = pd.Series(forecast_values, index=future_indices, name=f'{col}_Forecast')

        # Combine the forecasted values with the original data
        df.loc[future_indices, col] = forecast_df
    
    # Reset index
    df.index = original_index

    return df
    
    
# Define function to extract borders of South Sudan
def extract_regions(admin0_path=ADMIN0_PATH,
                    admin1_path=ADMIN1_PATH,
                    abyei_region_path=ABYEI_PATH):
    """
    Extract South Sudan regions for use in clipping tif files.

    Parameters:
    - admin0_path: str
        Path to top-level administrative boundaries of South Sudan (excluding the Abyei region).
    - admin1_path: str
        Path to state-level administrative boundaries of South Sudan (excluding the Abyei region).
    - abyei_region_path: str
        Path to Abyei region administrative boundaries.

    Returns:
    - ssd_gdf: GeoDataFrame
        A polygon object containing the administrative boundaries of South Sudan, including the Abyei region.
    """
    
    # Load shapefiles
    abyei_region = gpd.read_file(abyei_region_path)
    admin0 = gpd.read_file(admin0_path)
    admin1 = gpd.read_file(admin1_path)
    
    # Ensure both layers have the same coordinate reference system (CRS)
    if abyei_region.crs != admin0.crs:
        admin0 = admin0.to_crs(abyei_region.crs)
    
    # Use geopandas overlay with 'union' operation and set keep_geom_type=False to retain all geometries
    ssd_gdf = gpd.overlay(abyei_region, admin0, how='union', keep_geom_type=False)
    
    # Dissolve into a single polygon
    ssd_gdf = ssd_gdf.dissolve()
    
    # Check if CRS is geographic (latitude/longitude), and reproject if needed
    if ssd_gdf.crs.is_geographic:
        ssd_gdf = ssd_gdf.to_crs(epsg=3395)  # Example projected CRS (World Mercator)
    
    # Simplify the geometry (adjust the tolerance as needed)
    ssd_gdf['geometry'] = ssd_gdf.geometry.simplify(tolerance=1000)  # Use meters in projected CRS
    
    # Ensure geometry is valid (fix potential topology issues)
    ssd_gdf['geometry'] = ssd_gdf.buffer(10)  # Small buffer in meters to clean geometry
    
    # Reproject back to original CRS if needed
    ssd_gdf = ssd_gdf.to_crs(admin0.crs)
    
    # Rename columns
    abyei_region.rename({'admin2Name': 'region'}, axis=1, inplace=True)
    admin1.rename({'ADM1_EN': 'region'}, axis=1, inplace=True)
    ssd_gdf.rename({'ADM0_EN': 'region'}, axis=1, inplace=True)
    
    # Ensure both have the same CRS
    if abyei_region.crs != admin1.crs:
        abyei_region = abyei_region.to_crs(admin1.crs)
    
    # Make sure column names align (adjust as needed)
    common_columns = [col for col in admin1.columns if col in abyei_region.columns and col in ssd_gdf.columns]
    
    # Append Abyei as a new row
    regions_gdf = gpd.GeoDataFrame(pd.concat([ssd_gdf[common_columns], admin1[common_columns], abyei_region[common_columns]], ignore_index=True))
    
    return regions_gdf
    
    
# Define function to mask regions
def mask_regions(gdf, data):
    """
    Mask tif files using a polygon. 

    Parameters:
    - gdf: GeoDataFrame
        Polygon with target geometry for masking.
    - data: array
        3D array with values to be masked.
    Uses static reference grid config derived from
    data/downloads/inundation_masks_modis/20020801.tif (catchment-cropped).

    Returns:
    - masked_regions: array
        A 3D array of regions with areas outside the target polygon set to nan.
    """
    
    ref_crs = MASK_REGIONS_REF_CRS

    # Affine transform tuple: (a, b, c, d, e, f)
    ref_transform = MASK_REGIONS_REF_TRANSFORM
    ref_shape = MASK_REGIONS_REF_SHAPE
    
    # Ensure gdf boundary matches CRS of raster
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)
    
    # Rasterize gdf boundary: Inside = 1, Outside = 0
    gdf_mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,  # Outside polygon
        all_touched=True,  # Ensures full coverage
        dtype=np.uint8
    )
    
    # Ensure the mask has the same shape as the data
    if data[0].shape != gdf_mask.shape:
        raise ValueError(f"Shape mismatch: Inundation {data[0].shape[1:]} vs Mask {gdf_mask.shape}")
    
    # Apply the mask: Set values outside SSD to NaN
    masked_regions = np.where(gdf_mask == 1, data, np.nan)
    
    return masked_regions