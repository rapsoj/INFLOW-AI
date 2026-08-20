import os
import glob

from .. import cleaning_utils
from ..config import get_cfg

from data.stats import gridded_stats
from processing.data_cleaning.download_tamsat.tamsat_download_extract_api import download, extract

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import netCDF4 as nc

import h5py

from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOWNLOADS_ROOT = get_cfg("paths.downloads.root", "data/downloads")
GR_MOISTURE_DOWNLOAD_PATH = get_cfg("paths.downloads.tamsat_sm_daily", "data/downloads/tamsat/soil_moisture/data/v2.3.1/daily")
EXTRACTED_DOMAIN_PATH = get_cfg("paths.downloads.extracted_domain", "data/downloads/extracted_data/domain")
GR_MOISTURE_DEKADS_PATH = get_cfg("paths.downloads.tamsat_sm_dekads", "data/downloads/tamsat/soil_moisture/dekads")
GR_MOISTURE_TEMPORAL_PATH = cleaning_utils.get_target_historic_path("gridded_moisture_temporal.csv")
GR_MOISTURE_H5_PATH = cleaning_utils.get_target_historic_path("gridded_moisture.h5")
CATCHMENTS_PATH = get_cfg("paths.maps.catchments", "data/maps/inflow_catchments/INFLOW_all_cmts.shp")
TAMSAT_SM_VERSION = str(get_cfg("runtime.tamsat.sm_version", "2.3.1"))


def reproject_moisture(moisture_ds, target_crs, target_transform, target_width, target_height):
	"""Reproject and resample moisture data."""
	moisture_data = moisture_ds.read(1)
	moisture_transform = moisture_ds.transform
	moisture_crs = moisture_ds.crs

	reprojected_moisture = np.empty((target_height, target_width), dtype=moisture_data.dtype)
	reproject(
		source=moisture_data,
		destination=reprojected_moisture,
		src_transform=moisture_transform,
		src_crs=moisture_crs,
		dst_transform=target_transform,
		dst_crs=target_crs,
		resampling=Resampling.bilinear,
	)
	return reprojected_moisture


def extract_date_from_filename(filename):
	"""Extract date from moisture tif filename."""
	return cleaning_utils.extract_date_from_tif_filename(filename)


def download_gridded_moisture(dates_list, download_path):
	"""Download gridded moisture data for the specified dates."""
	try:
		download(
			{
				"product": "sm",
				"timestep": "daily",
				"resolution": 0.25,
				"start_date": dates_list[0],
				"end_date": dates_list[-1],
				"version": TAMSAT_SM_VERSION,
				"localdata_dir": download_path,
			}
		)
	except Exception as e:
		print(f"Error occurred while downloading TAMSAT data: {e}")


def extract_gridded_moisture(dates_list, download_path):
	"""Extract gridded moisture data for the specified dates into one domain file."""
	try:
		extract(
			{
				"product": "sm",
				"extract_type": "domain",
				"N": 15.837321509670957,
				"S": -4.029166662242848,
				"W": 23.424907051000087,
				"E": 36.30367723700005,
				"timestep": "daily",
				"resolution": 0.25,
				"start_date": dates_list[0],
				"end_date": dates_list[-1],
				"version": TAMSAT_SM_VERSION,
				"localdata_dir": download_path,
			}
		)
	except Exception as e:
		raise RuntimeError(f"Failed to extract gridded moisture data: {e}") from e


def _validate_moisture_netcdf_file(netcdf_path):
	"""Validate a domain moisture NetCDF file and raise if unreadable/corrupt."""
	required_variables = {"lat", "lon", "sm_c4grass", "time"}
	try:
		with nc.Dataset(netcdf_path, mode="r") as ds:
			missing = required_variables - set(ds.variables.keys())
			if missing:
				raise RuntimeError(f"Missing required NetCDF variables: {sorted(missing)}")

			_ = ds.variables["lat"][:1]
			_ = ds.variables["lon"][:1]
			_ = ds.variables["time"][:1]
			_ = ds.variables["sm_c4grass"][0:1, :, :]
	except Exception as e:
		raise RuntimeError(f"Invalid moisture NetCDF file '{netcdf_path}': {e}") from e


def _moisture_netcdf_date_range(netcdf_path):
	with nc.Dataset(netcdf_path, mode="r") as ds:
		time_var = ds.variables["time"]
		dates = nc.num2date(
			time_var[:],
			units=time_var.units,
			calendar=getattr(time_var, "calendar", "standard"),
			only_use_cftime_datetimes=False,
			only_use_python_datetimes=False,
		)
	return pd.Timestamp(dates[0]).normalize(), pd.Timestamp(dates[-1]).normalize()


def _get_latest_valid_moisture_netcdf(extract_folder, required_start=None, required_end=None):
	"""Return most recent readable moisture NetCDF covering the required range."""
	list_of_files = sorted(
		glob.glob(os.path.join(os.getcwd(), extract_folder, "*.nc")),
		key=os.path.getctime,
		reverse=True,
	)
	if not list_of_files:
		raise RuntimeError(f"No extracted NetCDF files found in '{extract_folder}'.")

	errors = []
	for path in list_of_files:
		try:
			_validate_moisture_netcdf_file(path)
			file_start, file_end = _moisture_netcdf_date_range(path)
			if required_start is not None and file_start > pd.Timestamp(required_start):
				raise RuntimeError(f"coverage starts at {file_start.date()}, before required {pd.Timestamp(required_start).date()}")
			if required_end is not None and file_end < pd.Timestamp(required_end):
				raise RuntimeError(f"coverage ends at {file_end.date()}, before required {pd.Timestamp(required_end).date()}")
			return path
		except Exception as e:
			errors.append(f"{os.path.basename(path)}: {e}")

	raise RuntimeError(
		"No valid extracted moisture NetCDF file found. Validation errors: " + " | ".join(errors)
	)


def get_historic_dates(data_path=GR_MOISTURE_TEMPORAL_PATH):
	"""Get list of historic dates from temporal moisture data."""
	try:
		gridded_moisture_temporal = pd.read_csv(data_path)
		if "date" in gridded_moisture_temporal.columns:
			historic_dates = gridded_moisture_temporal["date"].astype(str).tolist()
		else:
			historic_dates = gridded_moisture_temporal.index.astype(str).tolist()
		return historic_dates
	except FileNotFoundError:
		logging.info(f"Historic moisture temporal data not found at {data_path}; bootstrapping.")
		return []


def download_new_gridded_moisture(download_folder, target_product=None):
	"""Download gridded moisture for dates not already downloaded."""
	target_product = cleaning_utils.resolve_target_product(target_product)
	download_path_full = os.path.join(os.getcwd(), download_folder)
	current_date_str = datetime.now().strftime("%Y-%m-%d")
	historic_dates = get_historic_dates()
	has_historic_temporal = bool(historic_dates)

	if has_historic_temporal:
		last_date = datetime.strptime(historic_dates[-1], "%Y-%m-%d").strftime("%Y-%m-%d")
	else:
		last_date = cleaning_utils.get_target_start_date(target_product=target_product)

	new_dates = cleaning_utils.get_dates_of_interest(
		start_date_str=last_date,
		end_date_str=current_date_str,
		target_product=target_product,
	)

	local_product_path = os.path.join(os.getcwd(), GR_MOISTURE_DOWNLOAD_PATH)
	local_dates = cleaning_utils.get_local_download_dates(local_product_path)
	missing_dates = [d for d in new_dates if d not in local_dates]
	required_start = min(local_dates) if local_dates else last_date
	required_end = max(local_dates) if local_dates else current_date_str

	if missing_dates:
		download_range = [missing_dates[0], missing_dates[-1]]
		download_gridded_moisture(download_range, download_path_full)
		# Extraction must include the full cached archive, not only newly missing days.
		extract_range = [required_start, required_end] if local_dates else download_range
		extract_gridded_moisture(extract_range, download_folder)
		_get_latest_valid_moisture_netcdf(EXTRACTED_DOMAIN_PATH, required_start, required_end)
	else:
		try:
			_get_latest_valid_moisture_netcdf(EXTRACTED_DOMAIN_PATH, required_start, required_end)
		except RuntimeError:
			if local_dates:
				extract_gridded_moisture([min(local_dates), max(local_dates)], download_folder)
				_get_latest_valid_moisture_netcdf(EXTRACTED_DOMAIN_PATH, required_start, required_end)
			else:
				raise


def group_dates_by_target_period(dates, target_product="modis"):
	"""Group dates into target-product windows."""
	return cleaning_utils.group_dates_by_target_period(dates, target_product=target_product)


def export_decadal_geotiffs(extract_folder, output_folder, target_product="modis"):
	"""Export moisture grouped by target period into GeoTIFF files."""
	os.makedirs(output_folder, exist_ok=True)

	files = glob.glob(os.path.join(output_folder, "*"))
	for file in files:
		try:
			os.remove(file)
		except Exception as e:
			print(f"Error deleting {file}: {e}")

	latest_file = _get_latest_valid_moisture_netcdf(extract_folder)

	# Stream from NetCDF in small slices so the full time cube is never loaded.
	with nc.Dataset(latest_file, mode="r") as moisture_grid:
		lats = moisture_grid.variables["lat"][:]
		lons = moisture_grid.variables["lon"][:]
		times = moisture_grid.variables["time"][:]
		moisture_var = moisture_grid.variables["sm_c4grass"]

		first_date = datetime.strptime(latest_file[-24:-14], "%Y-%m-%d")
		dates = [(first_date + timedelta(days=int(i))) for i in times]

		print("--- Gridded moisture data loaded ---")

		date_groups, grouped_indices = group_dates_by_target_period(dates, target_product=target_product)

		lon_min = lons.min()
		lat_max = lats.max()
		pixel_size_x = abs(lons[1] - lons[0])
		pixel_size_y = abs(lats[1] - lats[0])
		transform = from_origin(lon_min, lat_max, pixel_size_x, pixel_size_y)

		for group, indices in tqdm(
			zip(date_groups, grouped_indices),
			total=len(date_groups),
			desc="Exporting decadal averages",
		):
			indices = list(indices)
			if not indices:
				continue

			sum_2d = None
			for idx in indices:
				arr_2d = np.array(moisture_var[idx, :, :], dtype=np.float32, copy=False)
				if sum_2d is None:
					sum_2d = np.zeros_like(arr_2d, dtype=np.float32)
				sum_2d += arr_2d

			decadal_avg = sum_2d / float(len(indices))
			if lats[0] < lats[-1]:
				decadal_avg = np.flipud(decadal_avg)

			first_dekad_str = group[0].strftime("%Y%m%d")
			output_file = os.path.join(output_folder, f"moisture_decadal_{first_dekad_str}.tif")

			with rasterio.open(
				output_file,
				"w",
				driver="GTiff",
				height=decadal_avg.shape[0],
				width=decadal_avg.shape[1],
				count=1,
				dtype=decadal_avg.dtype,
				crs="EPSG:4326",
				transform=transform,
			) as dst:
				dst.write(decadal_avg, 1)

			period_label = "VIIRS half-month" if target_product == "viirs" else "MODIS dekad"
			print(f"Exported {period_label} GeoTIFF for {first_dekad_str}")


def crop_historic_data(file_path, temporal_data_path):
	"""Crop temporal CSV length if HDF5 is shorter; rebuild HDF5 if longer."""
	hist = pd.read_csv(temporal_data_path)
	new_len = len(hist)

	with h5py.File(file_path, "r+") as f:
		dset_name = list(f.keys())[0]
		dset = f[dset_name]
		current_len = dset.shape[0]

		if current_len < new_len:
			hist.iloc[:current_len].to_csv(temporal_data_path, index=False)

		elif current_len > new_len:
			print(f"Cropping from {current_len} to {new_len} timesteps...")
			cropped = dset[:new_len]
			f.close()
			os.remove(file_path)

			with h5py.File(file_path, "w") as newf:
				newf.create_dataset(
					dset_name,
					data=cropped,
					maxshape=(None, *dset.shape[1:]),
					chunks=True,
					dtype=dset.dtype,
				)
			print("File truncated and recreated with cropped data.")
		else:
			print("No cropping needed. Temporal lengths already match.")


def process_new_gridded_moisture(moisture_dekads_folder):
	"""Create date-aligned moisture tif list for downstream streaming processing."""
	moisture_dekads_files = [f for f in os.listdir(moisture_dekads_folder) if f.endswith(".tif") and not f.endswith("(1).tif")]

	if not moisture_dekads_files:
		logging.info("No grouped moisture GeoTIFFs available to process yet.")
		return [], []

	moisture_dekads_files_new = glob.glob(os.path.join(moisture_dekads_folder, "*"))

	moisture_dates = [extract_date_from_filename(f) for f in moisture_dekads_files_new]
	moisture_dates = [d for d in moisture_dates if d is not None]

	dates = pd.to_datetime([date.strftime("%Y-%m-%d") for date in moisture_dates])
	dates_df = pd.DataFrame({"date": dates}).sort_values("date").reset_index()
	sorted_dates = list(dates_df["date"])
	moisture_df = pd.DataFrame({"moisture_file": moisture_dekads_files_new, "moisture_date": moisture_dates}).sort_values("moisture_date").reset_index()

	aligned_df = pd.merge(dates_df, moisture_df, left_on="date", right_on="moisture_date", how="left").sort_values("moisture_date").reset_index()

	missing_moisture_dates = aligned_df[aligned_df["moisture_file"].isna()]["date"]
	if not missing_moisture_dates.empty:
		print(f"Warning: Missing moisture data for {len(missing_moisture_dates)} dates.")

	aligned_moisture_files = aligned_df["moisture_file"].tolist()

	return aligned_moisture_files, sorted_dates


def process_single_moisture_tif(moisture_tif_path, catchments):
	"""Align and mask a single moisture GeoTIFF onto shared reference grid."""
	with rasterio.open(moisture_tif_path) as moisture_ds:
		resampled_moisture = cleaning_utils.align_and_mask_raster_to_reference_grid(
			src=moisture_ds,
			mask_gdf=catchments,
			src_band=1,
			dst_fill=0,
			resampling=Resampling.bilinear,
		)
	if resampled_moisture is None:
		raise RuntimeError(f"Reprojection returned None for file: {moisture_tif_path}")
	return resampled_moisture.astype(np.float32, copy=False)


def update_gridded_moisture(
	download_folder=DOWNLOADS_ROOT,
	download_path=GR_MOISTURE_DOWNLOAD_PATH,
	extract_folder=EXTRACTED_DOMAIN_PATH,
	dekads_path=GR_MOISTURE_DEKADS_PATH,
	temporal_data_path=GR_MOISTURE_TEMPORAL_PATH,
):
	"""Combine newly downloaded gridded moisture with existing data."""
	try:
		target_product = cleaning_utils.resolve_target_product(None)
		local_dates = cleaning_utils.get_local_download_dates(os.path.join(os.getcwd(), download_path))
		needs_rebuild = False
		if os.path.exists(GR_MOISTURE_H5_PATH) and os.path.exists(temporal_data_path):
			with h5py.File(GR_MOISTURE_H5_PATH, "r") as hdf:
				dset = hdf.get("moisture")
				if dset is None or dset.shape[0] == 0:
					needs_rebuild = True
				else:
					sample_indices = np.linspace(0, dset.shape[0] - 1, min(3, dset.shape[0]), dtype=int)
					needs_rebuild = not any(np.any(dset[index] != 0) for index in sample_indices)
			if needs_rebuild:
				logging.info("Moisture H5 contains no non-zero samples; rebuilding H5 and CSV.")

		if local_dates and os.path.exists(temporal_data_path):
			historic_dates = get_historic_dates(temporal_data_path)
			if historic_dates and min(local_dates) < min(historic_dates):
				needs_rebuild = True
				logging.info(
					"Moisture history starts at %s but local archive starts at %s; rebuilding H5 and CSV.",
					min(historic_dates),
					min(local_dates),
				)
			if needs_rebuild:
				for path in (GR_MOISTURE_H5_PATH, temporal_data_path):
					if os.path.exists(path):
						os.remove(path)

		if os.path.exists(GR_MOISTURE_H5_PATH) and os.path.exists(temporal_data_path):
			crop_historic_data(
				file_path=GR_MOISTURE_H5_PATH,
				temporal_data_path=temporal_data_path,
			)

		download_new_gridded_moisture(download_folder, target_product=target_product)

		dekads_path_full = os.path.join(os.getcwd(), dekads_path)
		export_decadal_geotiffs(extract_folder, dekads_path_full, target_product=target_product)
		aligned_files, dates = process_new_gridded_moisture(dekads_path_full)
		historic_dates = get_historic_dates()

		pending = []
		for i in range(len(dates)):
			date_str = dates[i].strftime("%Y-%m-%d")
			tif_name = aligned_files[i]
			if date_str in historic_dates:
				continue
			if pd.isna(tif_name):
				logging.warning(f"Skipping missing moisture file for date {date_str}")
				continue
			pending.append((os.path.join(dekads_path_full, tif_name), date_str))

		if len(pending) == 0:
			logging.info("No new files to process.")
		else:
			regions_gdf = cleaning_utils.extract_regions()
			catchments = gpd.read_file(CATCHMENTS_PATH)

			hdf_mode = "a" if os.path.exists(GR_MOISTURE_H5_PATH) else "w"

			temporal_rows = []
			with h5py.File(GR_MOISTURE_H5_PATH, hdf_mode) as hdf:
				dset = hdf.get("moisture")

				for tif_path, date_str in tqdm(pending, desc="Streaming moisture into H5"):
					try:
						moisture_2d = process_single_moisture_tif(tif_path, catchments)
					except Exception as e:
						raise RuntimeError(f"Failed processing moisture tif '{tif_path}': {e}") from e

					if dset is None:
						dset = hdf.create_dataset(
							"moisture",
							shape=(0, moisture_2d.shape[0], moisture_2d.shape[1]),
							maxshape=(None, moisture_2d.shape[0], moisture_2d.shape[1]),
							chunks=(1, moisture_2d.shape[0], moisture_2d.shape[1]),
							dtype=np.float32,
						)

					current_len = dset.shape[0]
					dset.resize(current_len + 1, axis=0)
					dset[current_len] = moisture_2d

					total_cells = moisture_2d.shape[0] * moisture_2d.shape[1]
					row = {
						"date": date_str,
						"moisture": float(np.nansum(moisture_2d)),
					}
					moisture_3d = moisture_2d[np.newaxis, :, :]
					for i in range(len(regions_gdf)):
						region_data = regions_gdf.iloc[[i]]
						region_code = gridded_stats.region_to_code_dict[region_data["region"].values[0]]
						region_area = cleaning_utils.mask_regions(region_data, moisture_3d)
						valid_cells = total_cells - np.sum(np.isnan(region_area[0]))
						row[f"moisture_{region_code}"] = (
							float(np.nansum(region_area)) / valid_cells if valid_cells > 0 else np.nan
						)

					temporal_rows.append(row)

				logging.info(f"Updated moisture dataset shape: {dset.shape}")

			moisture_temporal = pd.DataFrame(temporal_rows)
			if os.path.exists(temporal_data_path):
				moisture_temporal_historic = pd.read_csv(temporal_data_path)
			else:
				moisture_temporal_historic = pd.DataFrame(columns=moisture_temporal.columns)
			moisture_temporal_new = pd.concat([moisture_temporal_historic, moisture_temporal], ignore_index=True)

			moisture_temporal_new["date"] = pd.to_datetime(moisture_temporal_new["date"], errors="coerce")
			moisture_temporal_new = moisture_temporal_new.dropna(subset=["date"])
			moisture_temporal_new = moisture_temporal_new.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
			moisture_temporal_new.to_csv(temporal_data_path, index=False)

	except Exception as e:
		logging.error(f"Error processing moisture data: {e}")
		raise
