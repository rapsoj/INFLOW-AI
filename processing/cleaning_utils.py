# Import data manipulation libraries
import os
import calendar
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


def align_and_mask_raster_to_reference_grid(
    src,
    mask_gdf,
    src_band=1,
    dst_fill=0,
    resampling=Resampling.nearest,
):
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
    - resampling: rasterio.warp.Resampling
        Resampling method used during reprojection.

    Returns:
    - ndarray
        2D aligned+masked array in the static reference grid.
    """
    aligned = align_raster_to_reference_grid(
        src=src,
        src_band=src_band,
        dst_fill=dst_fill,
        resampling=resampling,
    )
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


def resolve_target_product(target_product=None):
    """Resolve target product from explicit argument or config."""
    resolved = (target_product or get_cfg("runtime.target_product", "modis")).lower().strip()
    if resolved not in {"modis", "viirs"}:
        raise ValueError(f"Invalid target_product '{resolved}'. Expected 'modis' or 'viirs'.")
    return resolved


<<<<<<< HEAD
def get_target_historic_path(filename, target_product=None):
    """Return a historic output path inside the selected product-aligned folder."""
    target_product = resolve_target_product(target_product)
    historic_root = get_cfg("paths.historic.root", "data/historic")
    return os.path.join(historic_root, f"{target_product}-aligned", filename)


=======
>>>>>>> origin/main
def _target_temporal_candidates(target_product):
    """Return candidate temporal CSV paths for a target product in priority order."""
    target_product = resolve_target_product(target_product)

    if target_product == "viirs":
        return [
            get_cfg("paths.historic.viirs_temporal", "data/historic/inundation_viirs_temporal.csv"),
            "data/historic/viirs_inundation_temporal.csv",
            "data/historic/inundation_viirs_temporal.csv",
        ]

    return [
        get_cfg("paths.historic.modis_temporal", "data/historic/inundation_modis_temporal.csv"),
        get_cfg("paths.historic.inundation_temporal", "data/historic/inundation_temporal.csv"),
        "data/historic/inundation_modis_temporal.csv",
    ]


def get_target_temporal_path(target_product=None):
    """Get the best available temporal CSV path for the selected target product."""
    candidates = _target_temporal_candidates(target_product)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


def get_target_start_date(target_product=None):
    """
    Get the first available date from the selected target product temporal CSV.

    For VIIRS data this prefers `period_start` and falls back to `date`.
    For MODIS this prefers `date`.
    """
    target_product = resolve_target_product(target_product)
    temporal_path = get_target_temporal_path(target_product)

    fallback = "2012-02-01" if target_product == "viirs" else "2002-07-01"
    if not temporal_path or not os.path.exists(temporal_path):
        return fallback

    try:
        df = pd.read_csv(temporal_path)
    except Exception:
        return fallback

    date_candidates = []
    if target_product == "viirs" and "period_start" in df.columns:
        date_candidates.append(pd.to_datetime(df["period_start"], errors="coerce"))
    if "date" in df.columns:
        date_candidates.append(pd.to_datetime(df["date"], errors="coerce"))

    if not date_candidates:
        return fallback

    combined = pd.concat(date_candidates, axis=0).dropna()
    if combined.empty:
        return fallback

    return combined.min().strftime("%Y-%m-%d")


def get_dates_of_interest(start_date_str=None, end_date_str=None, target_product=None):
    """
    Generate a list of target-aligned dates between start_date_str and end_date_str.

    Parameters:
        start_date_str (str | None): Start date in 'YYYY-MM-DD'. If None, inferred from target product temporal CSV.
        end_date_str (str | None): End date in 'YYYY-MM-DD'. Defaults to today if not provided.
        target_product (str | None): Either 'modis' or 'viirs'. If None, read from config.

    Returns:
        list: A list of dates (in 'YYYY-MM-DD' format) aligned to target cadence.
    """
    target_product = resolve_target_product(target_product)

    if start_date_str is None:
        start_date_str = get_target_start_date(target_product=target_product)

    valid_days = {1, 16} if target_product == "viirs" else {1, 11, 21}

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
        # MODIS cadence: 1, 11, 21. VIIRS cadence: 1, 16.
        if current_date.day in valid_days:
            dates_of_interest.append(current_date.strftime('%Y-%m-%d'))
        # Move to the next day
        current_date += timedelta(days=1)

    return dates_of_interest


def extract_date_from_tif_filename(filename):
    """
    Extract a date from a tif filename by searching for an 8-digit YYYYMMDD token.

    Returns:
        pandas.Timestamp | None
    """
    if not isinstance(filename, str) or not filename.lower().endswith('.tif'):
        return None

    digits = ''.join(ch for ch in filename if ch.isdigit())
    for i in range(0, max(0, len(digits) - 7)):
        token = digits[i:i + 8]
        try:
            return pd.to_datetime(token, format='%Y%m%d')
        except ValueError:
            continue
    return None


def get_local_download_dates(download_path_full):
    """Extract available local daily dates from downloaded files under a folder tree."""
    local_dates = set()
    if not os.path.exists(download_path_full):
        return local_dates

    for root, _, files in os.walk(download_path_full):
        for fname in files:
            digits = ''.join(ch for ch in fname if ch.isdigit())
            for i in range(0, max(0, len(digits) - 7)):
                token = digits[i:i + 8]
                try:
                    dt = datetime.strptime(token, '%Y%m%d')
                    local_dates.add(dt.strftime('%Y-%m-%d'))
                    break
                except ValueError:
                    continue
    return local_dates


def group_dates_by_target_period(dates, target_product='modis'):
    """
    Group daily dates into target-product windows.

    MODIS windows: day 1-10, 11-20, 21-end.
    VIIRS windows: day 1-15, 16-end.

    Returns:
        tuple[list[list[datetime]], list[list[int]]]
    """
    target_product = resolve_target_product(target_product)

    period_buckets = {}
    for idx, date in enumerate(dates):
        if target_product == 'viirs':
            start_day = 1 if date.day <= 15 else 16
        else:
            if date.day <= 10:
                start_day = 1
            elif date.day <= 20:
                start_day = 11
            else:
                start_day = 21

        key = datetime(date.year, date.month, start_day)
        period_buckets.setdefault(key, []).append(idx)

    date_groups = []
    grouped_indices = []
    for period_start, indices in sorted(period_buckets.items()):
        _, last_day = calendar.monthrange(period_start.year, period_start.month)

        if target_product == 'viirs':
            expected = 15 if period_start.day == 1 else (last_day - 15)
        else:
            if period_start.day in (1, 11):
                expected = 10
            else:
                expected = last_day - 20

        if len(indices) == expected:
            date_groups.append([dates[i] for i in indices])
            grouped_indices.append(indices)

    return date_groups, grouped_indices


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
        raise ValueError(f"Shape mismatch: data slice {data[0].shape} vs mask {gdf_mask.shape}")
    
    # Apply the mask: Set values outside SSD to NaN
    masked_regions = np.where(gdf_mask == 1, data, np.nan)
    
    return masked_regions