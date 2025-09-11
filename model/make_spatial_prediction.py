import os
import numpy as np
import pandas as pd
import h5py
import logging
from tqdm import tqdm
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from collections import defaultdict
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin
from rasterio.enums import Resampling as ResampleEnums
from shapely.geometry import Point, MultiPolygon
from matplotlib.patches import PathPatch
from pathlib import Path as FilePath
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
import zipfile
from processing.data_cleaning import process_inundation
from processing import cleaning_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def check_file_lengths(*file_paths):
    lengths = []
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            key = list(f.keys())[0]
            lengths.append(f[key].shape[0])
    return lengths


def generate_deployment_sequences_in_memory(
    inundation_file_path,
    rainfall_file_path,
    moisture_file_path,
    static_dir='data/maps',
    patch_size=64,
    stride=32,
    sequence_length=6,
    forecast_length=6,
    num_timesteps=1):
    """
    Generates deployment-ready spatial sequences (in memory) for the last available time slices.

    Returns:
        X: np.ndarray of shape (N, sequence_length, 5, patch_size, patch_size)
        indices: np.ndarray of shape (N, 3) where each row is (t, y, x)
    """

    inun_len, rain_len, moist_len = check_file_lengths(
        inundation_file_path, rainfall_file_path, moisture_file_path
    )

    if not (inun_len == rain_len == moist_len):
        logger.warning("Time series lengths do not match:")
        logger.warning(f"  - Inundation: {inun_len}")
        logger.warning(f"  - Rainfall:   {rain_len}")
        logger.warning(f"  - Moisture:   {moist_len}")
        logger.info(f"Using inundation length ({inun_len}) as reference.")
    T = inun_len

    # === Load static maps ===
    elevation = np.load(os.path.join(static_dir, 'elevation.npy'))  # (1, H, W)
    basin_attributes = np.load(os.path.join(static_dir, 'basin_attributes.npy'))[0:1]  # (1, H, W)

    with h5py.File(inundation_file_path, 'r') as inun_f, \
         h5py.File(rainfall_file_path, 'r') as rain_f, \
         h5py.File(moisture_file_path, 'r') as moist_f:

        inundation = inun_f['inundation']
        rainfall = rain_f['rainfall']
        moisture = moist_f['moisture']

        _, H, W = inundation.shape

        # Trim static maps
        elevation = elevation[:, :H, :W]
        basin_attributes = basin_attributes[:, :H, :W]

        # Define time window
        start_t = T - (num_timesteps + sequence_length + forecast_length)
        end_t = T - forecast_length

        y_positions = np.arange(0, H - patch_size + 1, stride)
        x_positions = np.arange(0, W - patch_size + 1, stride)

        num_time_steps = end_t - start_t
        num_spatial_patches = len(y_positions) * len(x_positions)
        total_samples = num_time_steps * num_spatial_patches

        logger.info(f"Generating {total_samples} input samples from time {start_t} to {end_t}...")

        # Preallocate in-memory arrays
        X = np.zeros((total_samples, sequence_length, 5, patch_size, patch_size), dtype=np.float32)
        indices = np.zeros((total_samples, 3), dtype=np.int32)

        sample_idx = 0

        t = end_t - 1  # Only the final timestep
        for y_start in y_positions:
            for x_start in x_positions:
                y_end = y_start + patch_size
                x_end = x_start + patch_size

                inun_seq = inundation[t:t+sequence_length, y_start:y_end, x_start:x_end]
                rain_seq = rainfall[t:t+sequence_length, y_start:y_end, x_start:x_end]
                moist_seq = moisture[t:t+sequence_length, y_start:y_end, x_start:x_end]

                elev_patch = elevation[:, y_start:y_end, x_start:x_end]
                basin_patch = basin_attributes[:, y_start:y_end, x_start:x_end]
                elev_seq = np.repeat(elev_patch, sequence_length, axis=0)
                basin_seq = np.repeat(basin_patch, sequence_length, axis=0)

                X_sample = np.stack([inun_seq, rain_seq, moist_seq, elev_seq, basin_seq], axis=1)
                X[sample_idx] = X_sample
                indices[sample_idx] = (t, y_start, x_start)
                sample_idx += 1

    logger.info(f"✅ In-memory sequence generation complete: {X.shape} samples")

    X = X[:sample_idx]
    indices = indices[:sample_idx]

    return X, indices


def masked_binary_focal_loss(gamma=2., alpha=0.25, border=4):
    def loss(y_true, y_pred):
        mask = tf.ones_like(y_true)
        mask = mask[:, border:-border, border:-border, :]
        y_true_crop = y_true[:, border:-border, border:-border, :]
        y_pred_crop = y_pred[:, border:-border, border:-border, :]

        y_pred_crop = K.clip(y_pred_crop, K.epsilon(), 1. - K.epsilon())
        p_t = y_true_crop * y_pred_crop + (1 - y_true_crop) * (1 - y_pred_crop)
        alpha_t = y_true_crop * alpha + (1 - y_true_crop) * (1 - alpha)
        return K.mean(mask * (-alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t)))
    return loss


def crop_borders(arr, border=4):
    return arr[:, border:-border, border:-border, :]


def load_trained_model(model_path, gamma=3.0, alpha=0.95):
    logger.info(f"Loading model from: {model_path}")
    model = load_model(
        model_path,
        custom_objects={"loss": masked_binary_focal_loss(gamma=gamma, alpha=alpha)}
    )
    logger.info("✅ Model loaded")
    return model


def load_deployment_inputs(X_path, indices_path):
    logger.info(f"Loading input sequences from: {X_path}")
    X = np.load(X_path)
    indices = np.load(indices_path)
    logger.info(f"✅ Loaded {X.shape[0]} samples for prediction")
    return X, indices


def predict_with_model(model, X, batch_size=32):
    logger.info("Running model prediction...")
    X_input = X.transpose(0, 1, 3, 4, 2)  # To (N, T, H, W, C)
    print("Making spatial predictions...")
    y_pred = model.predict(X_input, batch_size=batch_size, verbose=1)
    logger.info("✅ Prediction complete")
    return y_pred


def reconstruct_maps(y_pred, indices, full_shape, border=4):
    logger.info("Reconstructing full maps from predicted patches...")

    patch_size = y_pred.shape[1]
    y_pred_cropped = crop_borders(y_pred, border=border)
    cropped_size = patch_size - 2 * border

    time_to_patches = defaultdict(list)
    for i, (t, y, x) in enumerate(indices):
        patch = y_pred_cropped[i, :, :, 0]
        y0 = y + border
        x0 = x + border
        y1 = y0 + cropped_size
        x1 = x0 + cropped_size
        time_to_patches[t].append((patch, y0, y1, x0, x1))

    time_maps = {}
    for t, patches in time_to_patches.items():
        sum_map = np.zeros(full_shape, dtype=np.float32)
        count_map = np.zeros(full_shape, dtype=np.uint8)

        for patch, y0, y1, x0, x1 in patches:
            sum_map[y0:y1, x0:x1] += patch
            count_map[y0:y1, x0:x1] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_map = np.true_divide(sum_map, count_map)
            avg_map[count_map == 0] = np.nan
        time_maps[t] = avg_map

    logger.info(f"✅ Reconstructed {len(time_maps)} time-indexed maps")
    return time_maps


def run_inference_pipeline(
    model_path='model/spatial_model.keras',
    inundation_file_path='data/historic/inundation.h5',
    rainfall_file_path='data/historic/gridded_rainfall.h5',
    moisture_file_path='data/historic/gridded_moisture.h5',
    static_dir='data/maps',
    full_shape=(1125, 1204),
    border=4
):
    model = load_trained_model(model_path)

    X, indices = generate_deployment_sequences_in_memory(
        inundation_file_path,
        rainfall_file_path,
        moisture_file_path,
        static_dir=static_dir
    )

    y_pred = predict_with_model(model, X)
    maps = reconstruct_maps(y_pred, indices, full_shape, border=border)
    return maps


def load_spatial_ref(inundation_path="data/downloads/inundation_masks/20250211.tif",
                     catchments_path="data/maps/inflow_catchments/INFLOW_all_cmts.shp",
                     download_path='data/downloads/inundation_masks',
                     inundation_file="20250211.tif"):

      # Process the new TIF files
      with rasterio.open(inundation_path) as src:
          inundation = src.read(1)  # First band
          transform = src.transform
          crs = src.crs
          width = src.width
          height = src.height
      catchments = process_inundation.load_shapefile(catchments_path)
      catchments = process_inundation.reproject_to_raster_crs(catchments, inundation_path)

      # Process rasters and gather new data
      inundation_clipped, _, _ = process_inundation.process_and_clip_rasters([inundation_file], download_path, catchments)
      inundation = inundation_clipped[0]

      # Crop area to regions of interest
      regions_gdf = process_inundation.cleaning_utils.extract_regions()

      import numpy as np
      from rasterio.features import rasterize

      # Assign each region a unique integer ID
      regions_gdf = regions_gdf.copy()
      regions_gdf['region_id'] = np.arange(1, len(regions_gdf) + 1)

      # Prepare list of (geometry, value) tuples
      shapes = list(zip(regions_gdf.geometry, regions_gdf['region_id']))

      # Get transform and shape from raster
      height, width = inundation.shape
      regions_gdf = regions_gdf.to_crs(crs)

      # Reproject regions_gdf to WGS84 (EPSG:4326)
      regions_gdf = regions_gdf.to_crs(epsg=4326)

      # Raster reprojection to EPSG:4326
      from rasterio.warp import calculate_default_transform, reproject, Resampling

      dst_crs = 'EPSG:4326'
      dst_transform, dst_width, dst_height = calculate_default_transform(
          crs, dst_crs, width, height, *src.bounds)

      # Create an empty array for reprojected raster
      reprojected_inundation = np.empty(shape=(height, width), dtype=inundation.dtype)
      reproject(
          source=inundation,
          destination=reprojected_inundation,
          src_transform=src.transform,  # ✅ the original transform
          src_crs=crs,
          dst_transform=dst_transform,  # ✅ the calculated new one
          dst_crs=dst_crs,
          resampling=Resampling.nearest
      )

      # Return the new transform, CRS, and reprojected regions_gdf & raster array
      return dst_transform, dst_crs, regions_gdf, reprojected_inundation


def polygon_to_path_patch(polygon, hatch='///', **kwargs):
    """Convert a Shapely Polygon or MultiPolygon into a matplotlib PathPatch with hatching."""
    if isinstance(polygon, MultiPolygon):
        patches = []
        for poly in polygon.geoms:
            patches.append(polygon_to_path_patch(poly, hatch=hatch, **kwargs))
        return patches

    # Extract exterior and interior coordinates
    vertices = []
    codes = []

    # Exterior
    x, y = polygon.exterior.coords.xy
    verts = np.column_stack([x, y])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    vertices.extend(verts.tolist())

    # Interiors (holes)
    for interior in polygon.interiors:
        x, y = interior.coords.xy
        verts = np.column_stack([x, y])
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
        vertices.extend(verts.tolist())

    path = MplPath(vertices, codes)
    return PathPatch(path, facecolor='none', edgecolor='black', hatch=hatch, lw=0.0, alpha=0.4, **kwargs)


def align_mask(mask_array, src_transform, src_crs, metas, inundation_file):
    aligned = np.zeros((
        metas[inundation_file]["height"],
        metas[inundation_file]["width"]
        ), dtype=np.uint8)
    reproject(
        source=mask_array,
        destination=aligned,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=metas[inundation_file]["transform"],
        dst_crs=metas[inundation_file]['crs'],
        resampling=ResampleEnums.nearest
    )
    return aligned


def plot_flood_change_map(
    masks, current_extent, transform, crs, regions_gdf,
    inundation_clipped, metas, inundation_file='20250211.tif',
    title="Flood Change Map", region_name=None):

    # --- Align all masks ---
    current = align_mask((current_extent > 0).astype(np.uint8), metas[inundation_file]["transform"], metas[inundation_file]['crs'], metas, inundation_file)
    worst = align_mask((masks["Worst Case"] > 0).astype(np.uint8), metas[inundation_file]["transform"], metas[inundation_file]['crs'], metas, inundation_file)
    avg = align_mask((masks["Average Case"] > 0).astype(np.uint8), metas[inundation_file]["transform"], metas[inundation_file]['crs'], metas, inundation_file)
    best = align_mask((masks["Best Case"] > 0).astype(np.uint8), metas[inundation_file]["transform"], metas[inundation_file]['crs'], metas, inundation_file)

    # Use the transform from clipped raster metadata (not src!)
    out_meta = metas[inundation_file].copy()
    target_transform = out_meta["transform"]
    target_crs = out_meta["crs"]
    target_shape = inundation_clipped[0].shape

    # --- Change classification ---
    currently_flooded = current == 1
    currently_dry = current == 0

    no_change_dry = currently_dry & (worst == 0) & (avg == 0) & (best == 0)
    no_change_flooded = currently_flooded & (worst == 1) & (avg == 1) & (best == 1)
    very_likely_decrease = currently_flooded & (worst == 0) & (avg == 0) & (best == 0)
    likely_decrease = currently_flooded & (worst == 1) & (avg == 0) & (best == 0)
    possible_decrease = currently_flooded & (worst == 1) & (avg == 1) & (best == 0)
    very_likely_increase = currently_dry & (worst == 1) & (avg == 1) & (best == 1)
    likely_increase = currently_dry & (worst == 1) & (avg == 1) & (best == 0)
    possible_increase = currently_dry & (worst == 1) & (avg == 0) & (best == 0)

    # --- Color mapping ---
    colors = np.zeros((*current.shape, 4))
    colors[no_change_dry] = mcolors.to_rgba("#ffffff")
    colors[no_change_flooded] = mcolors.to_rgba("#d3d3d3")
    colors[possible_decrease] = mcolors.to_rgba("#f4a261")
    colors[likely_decrease] = mcolors.to_rgba("#e76f51")
    colors[very_likely_decrease] = mcolors.to_rgba("#9b2226")
    colors[possible_increase] = mcolors.to_rgba("#a8dadc")
    colors[likely_increase] = mcolors.to_rgba("#457b9d")
    colors[very_likely_increase] = mcolors.to_rgba("#1d3557")

    # --- Select region for boundary mask ---
    if region_name:
        selected_region = regions_gdf[regions_gdf['region'] == region_name]
        selected_region = selected_region.to_crs(target_crs)
        if selected_region.empty:
            raise ValueError(f"Region '{region_name}' not found in regions_gdf.")
        state_boundaries = selected_region
        country_boundary = selected_region
    else:
        state_boundaries = regions_gdf[regions_gdf["region_id"] != 1]
        country_boundary = regions_gdf[regions_gdf["region_id"] == 1]
    state_boundaries = state_boundaries.to_crs(target_crs)
    country_boundary = country_boundary.to_crs(target_crs)

    # --- Rasterize mask for region ---
    mask_geom = selected_region.geometry if region_name else country_boundary.geometry
    region_mask = rasterize(
        [(geom, 1) for geom in mask_geom],
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        dtype='uint8'
    )
    colors[region_mask == 0] = (1, 1, 1, 0)  # transparent

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 10))
    from rasterio.plot import plotting_extent

    extent = plotting_extent(
        np.zeros(target_shape),
        transform=target_transform
    )
    ax.imshow(colors, origin='upper', extent=extent)

    # --- Plot region boundaries ---
    state_boundaries.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.25)
    country_boundary.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.5)

    # --- Continue with legends, labels, points, hatching, etc. (unchanged) ---
    # Get Abyei geometry
    abyei_region = regions_gdf[regions_gdf["region"] == "Abyei Region"]
    abyei_region = abyei_region.to_crs(target_crs)

    # Only plot outline and hatching if full country or Abyei is selected
    show_abyei = region_name is None or region_name == "Abyei Region"

    if show_abyei:
        # Overlay dashed border
        for _, row in abyei_region.iterrows():
            g = row.geometry
            if g.geom_type == "MultiPolygon":
                for part in g.geoms:
                    x, y = part.exterior.xy
                    ax.plot(x, y, linestyle='--', color='white', linewidth=1.5)
            else:
                x, y = g.exterior.xy
                ax.plot(x, y, linestyle='--', color='white', linewidth=1.6)

        # Add hatching
        for _, row in abyei_region.iterrows():
            geom = row.geometry
            patch_or_patches = polygon_to_path_patch(geom, hatch='///')
            if isinstance(patch_or_patches, list):
                for patch in patch_or_patches:
                    ax.add_patch(patch)
            else:
                ax.add_patch(patch_or_patches)

    # --- Main legend for flood categories (bottom left) ---
    flood_legend_patches = [
        mpatches.Patch(color="#d3d3d3", label="No change (remains flooded)"),
        mpatches.Patch(color="#f4a261", label="Possible flood decrease (best-case only)"),
        mpatches.Patch(color="#e76f51", label="Likely flood decrease (average case)"),
        mpatches.Patch(color="#9b2226", label="Very likely flood decrease (all scenarios)"),
        mpatches.Patch(color="#a8dadc", label="Possible flood increase (worst-case only)"),
        mpatches.Patch(color="#457b9d", label="Likely flood increase (average case)"),
        mpatches.Patch(color="#1d3557", label="Very likely flood increase (all scenarios)")
    ]
    legend1 = ax.legend(
        handles=flood_legend_patches,
        loc="lower left",
        fontsize=10,
        frameon=False
    )
    ax.add_artist(legend1)  # ensure both legends show

    # --- Separate legend for population centres (bottom right) ---
    if region_name:
      pop_label = 'Population centre'
    else:
      pop_label = 'Population centre (≥10,000 people)'

    population_legend = Line2D(
        [0], [0],
        marker='o',
        color='black',
        markerfacecolor='white',
        markersize=8,
        linewidth=0,
        label=pop_label
    )
    msf_legend = Line2D(
        [0], [0],
        marker='+',
        color='red',
        markersize=10,
        linewidth=1,
        label='MSF project location'
    )

    legend2 = ax.legend(
        handles=[population_legend, msf_legend],
        loc="lower right",
        fontsize=9,
        frameon=False
    )

    # Add region labels
    for _, row in state_boundaries.iterrows():
        point = row.geometry.representative_point()
        txt = ax.annotate(
            text=row['region'],
            xy=(point.x, point.y + 0.15),
            ha='center', va='center',
            fontsize=8,
            color='black',
            alpha=0.9,
            fontstyle='italic'
        )
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=2, foreground="white")
        ])

    # Plot population centres
    pop_centres = gpd.read_file("data/maps/population_centres/hotosm_ssd_populated_places_points_shp.shp")
    pop_centres['name_en'] = np.where(pop_centres['name_en'].isna(), pop_centres['name'], pop_centres['name_en'])
    if pop_centres.crs != target_crs:
      pop_centres = pop_centres.to_crs(target_crs)
    pop_centres = pop_centres[pop_centres['population'].notna()]
    pop_centres = pop_centres[pop_centres['population'].astype(int) >= 10000]

    # Filter to selected region if applicable
    if region_name:
        pop_centres = gpd.sjoin(pop_centres, selected_region, how="inner", predicate="intersects")

    # Clip population centres to region
    if region_name:
        pop_centres_clipped = gpd.clip(pop_centres, selected_region)
    else:
        pop_centres_clipped = pop_centres

    # Plot only if non-empty
    if not pop_centres_clipped.empty:
        pop_centres_clipped.plot(ax=ax, markersize=20, color='white', edgecolor='black', linewidth=0.5, zorder=3)


    # --- Manually entered MSF project locations (as red plus signs, no labels) ---
    msf_locations = [
        ("Leer", 8.301, 30.107),
        ("Lankien", 8.519, 31.956),
        ("Bentiu", 9.239, 29.506),
        ("Renk", 11.746, 32.766),
        ("EGPAA (Boma & Maruwa)", 6.234, 34.193),
        ("Old Fangak", 9.0707, 30.8146),
        ("Aweil", 8.721, 27.276),
        ("Abyei", 9.628, 28.0616),
        ("Twic (Mayen Abun)", 9.1275, 28.11)
    ]

    # Plot MSF sites
    for _, lat, lon in msf_locations:
        point_geo = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(target_crs)
        x, y = point_geo.geometry.x.values[0], point_geo.geometry.y.values[0]

        if region_name:
            if not point_geo.intersects(selected_region.union_all()).iloc[0]:
                continue
        ax.plot(x, y, marker='+', color='red', markersize=7, zorder=5)

    # Add labels
    for _, row in pop_centres.iterrows():
        x, y = row.geometry.x, row.geometry.y
        label = row.get("name_en", "")
        if label:
            txt = ax.text(
                x + 0.02, y + 0.02,  # slight offset
                label,
                fontsize=6,
                color='black',
                zorder=4
            )
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=1.5, foreground="white")
            ])

    # Zoom to selected region
    if region_name:
        bounds = selected_region.total_bounds  # (minx, miny, maxx, maxy)
    else:
        bounds = country_boundary.total_bounds  # fallback to country-wide bounds

    # Compute width and height of bounds
    xmin, ymin, xmax, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin

    # Define padding as a percentage of width/height
    if region_name:
        x_pad_pct = 0.20
        y_pad_top_pct = 0.01
        y_pad_bottom_pct = 0.20
    else:
        x_pad_pct = 0.05
        y_pad_top_pct = 0.01
        y_pad_bottom_pct = 0.10

    # Convert percentages to absolute padding
    x_pad = width * x_pad_pct
    y_pad_top = height * y_pad_top_pct
    y_pad_bottom = height * y_pad_bottom_pct

    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad_bottom, ymax + y_pad_top)

    ax.axis('off')
    ax.set_title(title, size=16)
    ax.text(
    0.5, 1.01,  # x centered, y just under the main title
    "*Forecast of MODIS flood mask at 1km x 1km resolution",  # your subtitle text
    transform=ax.transAxes,
    ha='center',
    fontsize=10,
    style='italic'
    )

    plt.tight_layout()

    return fig


def load_latest_prediction_csv(base_path="predictions"):
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        raise FileNotFoundError("No subdirectories found in predictions folder.")

    latest_folder = max(subdirs, key=os.path.getmtime)
    folder_name = os.path.basename(latest_folder)
    csv_files = [f for f in os.listdir(latest_folder) if f.endswith('.csv')]

    if len(csv_files) != 1:
        raise ValueError("Expected one CSV file in latest prediction folder.")

    df = pd.read_csv(os.path.join(latest_folder, csv_files[0]), index_col=0)
    return df, folder_name


def generate_flood_mask(prob_map, flood_percent):
    valid_mask = ~np.isnan(prob_map)
    valid_probs = prob_map[valid_mask]
    n_flood_pixels = int(np.round(flood_percent * valid_probs.size))
    sorted_probs = np.sort(valid_probs)[::-1]
    threshold = sorted_probs[n_flood_pixels - 1] if n_flood_pixels > 0 else np.inf
    binary_mask = np.zeros_like(prob_map, dtype=np.uint8)
    binary_mask[(prob_map >= threshold) & valid_mask] = 1
    return binary_mask


def export_raster(mask, path, transform, crs):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(mask.astype('uint8'), 1)


def export_qgis_files(masks, current_extent, transform, crs, regions_gdf, folder_title,
                      metas, inundation_file='20250211.tif'):
  
    from tempfile import TemporaryDirectory

    zip_path = FilePath(f"predictions/{folder_title}/spatial_predictions/flood_prediction_spatial_data.zip")
    with TemporaryDirectory() as temp_dir:
        temp_path = FilePath(temp_dir)

        transform = metas[inundation_file]["transform"]
        crs = metas[inundation_file]["crs"]

        # Align and export rasters
        export_raster(
            align_mask((masks["Worst Case"] > 0).astype(np.uint8),
                      metas[inundation_file]["transform"],
                      metas[inundation_file]["crs"], metas, inundation_file),
            temp_path / "flood_scenario_worst.tif", transform, crs
        )

        export_raster(
            align_mask((masks["Average Case"] > 0).astype(np.uint8),
                      metas[inundation_file]["transform"],
                      metas[inundation_file]["crs"], metas, inundation_file),
            temp_path / "flood_scenario_average.tif", transform, crs
        )

        export_raster(
            align_mask((masks["Best Case"] > 0).astype(np.uint8),
                      metas[inundation_file]["transform"],
                      metas[inundation_file]["crs"], metas, inundation_file),
            temp_path / "flood_scenario_best.tif", transform, crs
        )

        export_raster(
            align_mask((current_extent > 0).astype(np.uint8),
                      metas[inundation_file]["transform"],
                      metas[inundation_file]["crs"], metas, inundation_file),
            temp_path / "current_extent.tif", transform, crs
        )

        regions_gdf.to_file(temp_path / "admin_boundaries.shp")

        pop_centres = gpd.read_file("data/maps/population_centres/hotosm_ssd_populated_places_points_shp.shp")
        pop_centres = pop_centres[pop_centres['population'].notna()]
        pop_centres = pop_centres[pop_centres['population'].astype(int) >= 10000]
        pop_centres = pop_centres.to_crs(crs)
        pop_centres.to_file(temp_path / "population_centres.shp")

        msf_data = [
            ("Leer", 8.301, 30.107),
            ("Lankien", 8.519, 31.956),
            ("Bentiu", 9.239, 29.506),
            ("Renk", 11.746, 32.766),
            ("EGPAA (Boma & Maruwa)", 6.234, 34.193),
            ("Old Fangak", 9.0707, 30.8146),
            ("Aweil", 8.721, 27.276),
            ("Abyei", 9.628, 28.0616),
            ("Twic (Mayen Abun)", 9.1275, 28.11)
        ]
        msf_gdf = gpd.GeoDataFrame(
            msf_data,
            columns=["name", "lat", "lon"],
            geometry=gpd.points_from_xy([x[2] for x in msf_data], [x[1] for x in msf_data]),
            crs="EPSG:4326"
        ).to_crs(crs)
        msf_gdf.to_file(temp_path / "msf_locations.shp")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_path.rglob("*"):
                zipf.write(file, arcname=file.name)

        print(f"Exported QGIS-ready files to: {zip_path.resolve()}")


def run_full_spatial_analysis():
    # Run inference
    maps = run_inference_pipeline()

    # Load prediction CSV
    pred_df, folder_title = load_latest_prediction_csv()

    # Determine example time index
    assert sorted(maps.keys())[-1] + 13 == len(pred_df), f"Mismatch in prediction timeline with historic data = {sorted(maps.keys())[-1] + 13} and prediction data = {len(pred_df)}"
    example_t = sorted(maps.keys())[-1]

    # Extract flood thresholds
    flood_percentages = {
        "Worst Case": pred_df.iloc[-1]['upper_bound_95'],
        "Average Case": pred_df.iloc[-1]['percent_inundation'],
        "Best Case": pred_df.iloc[-1]['lower_bound_95']
    }

    # Generate flood masks
    masks = {
        scenario: generate_flood_mask(maps[example_t], pct)
        for scenario, pct in flood_percentages.items()
    }

    # Load spatial reference and current extent
    with h5py.File('data/historic/inundation.h5', 'r') as f:
        current_extent = f['inundation'][-1]

    transform, crs, regions_gdf, _ = load_spatial_ref()

    # Load transformation reference
    inundation_path="data/downloads/inundation_masks/20250211.tif"
    catchments_path="data/maps/inflow_catchments/INFLOW_all_cmts.shp"
    download_path='data/downloads/inundation_masks'
    inundation_file="20250211.tif"

    # Process the new TIF files
    with rasterio.open(inundation_path) as src:
        inundation = src.read(1)  # First band
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
    catchments = process_inundation.load_shapefile(catchments_path)
    catchments = process_inundation.reproject_to_raster_crs(catchments, inundation_path)

    # Run the clipping function
    inundation_clipped, _, metas = process_inundation.process_and_clip_rasters(
        [inundation_file], download_path, catchments)

    # Create output directory
    output_dir = FilePath(f"predictions/{folder_title}/spatial_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot full country
    title = f"Predicted South Sudan Flood Extent Change from {folder_title[23:33]} to {folder_title[-10:]}\n"
    fig = plot_flood_change_map(masks, current_extent, transform, crs, regions_gdf,
                                inundation_clipped, metas, inundation_file='20250211.tif', title=title)
    fig.savefig(output_dir / f"south_sudan_spatial_prediction_{folder_title[23:33]}_to_{folder_title[-10:]}.png", dpi=300)
    plt.close(fig)

    # Plot regions
    for region in regions_gdf['region']:
        if region.strip().lower() == "south sudan":
            continue

        region_clean = "_".join(region.lower().split())
        title = f"Predicted {region} Flood Extent Change from {folder_title[23:33]} to {folder_title[-10:]}\n"

        try:
            fig = plot_flood_change_map(
                masks, current_extent, transform, crs, regions_gdf,
                inundation_clipped, metas, inundation_file='20250211.tif',
                region_name=region, title=title)
            filename = f"{region_clean}_spatial_prediction_{folder_title[23:33]}_to_{folder_title[-10:]}.png"
            fig.savefig(output_dir / filename, dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting {region}: {e}")

    # Export for QGIS
    export_qgis_files(masks, current_extent, transform, crs, regions_gdf, folder_title, metas, inundation_file='20250211.tif')