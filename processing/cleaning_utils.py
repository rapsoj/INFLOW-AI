# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import geospatial libraries
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
import geopandas as gpd

# Import machine learning libraries
from sklearn.linear_model import LinearRegression


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
def extract_regions(admin0_path='data/maps/admin_boundaries/ssd_admbnda_adm0_imwg_nbs_20230829.shp',
                    admin1_path='data/maps/admin_boundaries/ssd_admbnda_adm1_imwg_nbs_20230829.shp',
                    abyei_region_path='data/maps/abyei_region/ssd_admbnda_abyei_imwg_nbs_20180401.shp'):
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
def mask_regions(gdf, data, ref_path='data/downloads/inundation_masks/20241111.tif', catchments_path='data/maps/inflow_catchments/INFLOW_all_cmts.shp'):
    """
    Mask tif files using a polygon. 

    Parameters:
    - gdf: GeoDataFrame
        Polygon with target geometry for masking.
    - data: array
        3D array with values to be masked.
    - ref_path: str
        Path to reference file for aligning array.
    - catchments_path: str
        Path catchments of interest for cropping polygon.

    Returns:
    - masked_regions: array
        A 3D array of regions with areas outside the target polygon set to nan.
    """
    
    # Load the catchments shapefile using GeoPandas
    catchments = gpd.read_file(catchments_path)
    
    # Read the raster file to get the CRS and transform
    with rasterio.open(ref_path) as src:
        ref_crs = src.crs  # Get raster CRS
    
        # Reproject catchments to match raster CRS if needed
        if catchments.crs != ref_crs:
            catchments = catchments.to_crs(ref_crs)
        
        # Get the geometry of the catchments as a list of polygons
        catchment_geom = catchments.geometry.values  # This returns a list of geometries
        
        # Clip the raster using the catchment geometries
        ref_image, ref_transform = mask(src, catchment_geom, crop=True)
    
    # Ensure gdf boundary matches CRS of raster
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)
    
    # Rasterize gdf boundary: Inside = 1, Outside = 0
    gdf_mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=np.squeeze(ref_image, axis=0).shape,
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