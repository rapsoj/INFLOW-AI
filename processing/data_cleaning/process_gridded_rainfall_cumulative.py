# Import system libraries
import os

# Import cleaning utils
from .. import cleaning_utils
from ..config import get_cfg

# Import statistics
from data.stats import gridded_stats

# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime

# Import compression libraries
import h5py

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GR_RAINFALL_TEMPORAL_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall_temporal.csv")
GR_RAINFALL_H5_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall.h5")
GR_RAINFALL_CUM_TEMPORAL_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall_cumulative_temporal.csv")
GR_RAINFALL_CUM_H5_PATH = cleaning_utils.get_target_historic_path("gridded_rainfall_cumulative.h5")


def read_stats(region='all'):
    """
    Retained for compatibility with callers from older pipeline versions.
    """
    return 0.0, 1.0


def standardize_array(array, mean, std):
    """
    Standard scale the 3D array.

    Parameters:
        array (array): Array to be standard scaled.
        mean (float): Mean value for standardisation.
        std (float): Standard deviation value for standardisation.
    """
    # Apply standardization
    standardized_array = (array - mean) / std
    return standardized_array


def unstandardize_array(array, mean, std):
    """
    Undo standard scaling of the 3D array.

    Parameters:
        array (array): Array to be unscaled.
        mean (float): Mean value for unstandardisation.
        std (float): Standard deviation value for unstandardisation.
    """
    # Apply unstandardization
    unstandardized_array = array * std + mean
    return unstandardized_array
    
    
def get_historic_dates(data_path=GR_RAINFALL_CUM_TEMPORAL_PATH):
    """
    Get list of historic dates from pre-downloaded data.

    Parameters:
        data_path (str): Directory path of pre-downloaded temporal data.
    """
    try:
        gridded_rainfall_cumulative_temporal = pd.read_csv(data_path, index_col=0)
        historic_dates = gridded_rainfall_cumulative_temporal.index.tolist()
        return historic_dates
    except FileNotFoundError:
        logging.info(f"Historic cumulative rainfall temporal data not found at {data_path}; bootstrapping.")
        return []
        
        
def get_new_dates(rainfall_data_path=GR_RAINFALL_TEMPORAL_PATH,
                  cumulative_rainfall_data_path=GR_RAINFALL_CUM_TEMPORAL_PATH):
    """
    Get list of new dates from pre-downloaded data.

    Parameters:
        data_path (str): Directory path of pre-downloaded temporal data.
    """
    try:
        gridded_rainfall_temporal = pd.read_csv(rainfall_data_path, index_col=0)
        if not os.path.exists(cumulative_rainfall_data_path):
            return gridded_rainfall_temporal.index.tolist()

        gridded_rainfall_cumulative_temporal = pd.read_csv(cumulative_rainfall_data_path, index_col=0)
        new_dates = gridded_rainfall_temporal.iloc[len(gridded_rainfall_cumulative_temporal):].index.tolist()
        return new_dates
    except FileNotFoundError:
        logging.info("Cumulative rainfall temporal data not found; bootstrapping from rainfall data.")
        return []



def load_new_gridded_rainfall_data(temporal_data_path=GR_RAINFALL_TEMPORAL_PATH,
                                   cum_temporal_data_path=GR_RAINFALL_CUM_TEMPORAL_PATH,
                                   data_path=GR_RAINFALL_H5_PATH,
                                   cum_data_path=GR_RAINFALL_CUM_H5_PATH):
	"""
	Load new gridded rainfall data to be combined with cumulative sums.

	Parameters:
		temporal_data_path (str): Path to temporal gridded rainfall data.
		cum_temporal_data_path (str): Path to temporal gridded cumulative rainfall data.
		data_path (str): Path to gridded rainfall data.
		cum_data_path (str): Path to gridded cumulative rainfall data.
	"""

	# Identify the new recent historic gridded rainfall data
	if os.path.exists(cum_temporal_data_path) and os.path.exists(cum_data_path):
		len_old = len(pd.read_csv(cum_temporal_data_path))
	else:
		len_old = 0
	len_new = len(pd.read_csv(temporal_data_path))
	new_rainfall_indices = len_new - len_old
	if new_rainfall_indices <= 0:
	    return None, None
    
	# Open the new gridded rainfall data
	with h5py.File(data_path, 'r') as gridded_rainfall:
	    # Access the dataset (replace 'your_dataset' with the actual dataset name)
	    dataset = gridded_rainfall['rainfall']
	    
	    # Load the new rainfall grids
	    gridded_rainfall_new = dataset[-new_rainfall_indices:, :, :]

	# Open the last cumulative gridded rainfall data
	if len_old == 0:
		gridded_rainfall_cumulative_last = np.zeros(
			(1, *gridded_rainfall_new.shape[1:]), dtype=gridded_rainfall_new.dtype
		)
	else:
		with h5py.File(cum_data_path, 'r') as gridded_rainfall_cumulative:
			dataset = gridded_rainfall_cumulative['cumulative_rainfall']
			gridded_rainfall_cumulative_last = np.expand_dims(dataset[-1, :, :], axis=0)

	return gridded_rainfall_new, gridded_rainfall_cumulative_last
	
	
def crop_historic_data(file_path, temporal_data_path):
    """
    Crop or recreate the historic inundation HDF5 dataset to match the temporal CSV length.

    If the HDF5 dataset is longer than the CSV, the entire HDF5 file will be truncated and recreated.
    """

    # --- Load temporal data length only ---
    hist = pd.read_csv(temporal_data_path)
    new_len = len(hist)

    # --- Open HDF5 and check dataset length ---
    with h5py.File(file_path, "r+") as f:
        dset_name = list(f.keys())[0]
        dset = f[dset_name]
        current_len = dset.shape[0]

        if current_len < new_len:
            hist.iloc[:current_len].to_csv(temporal_data_path, index=False)

        elif current_len > new_len:
            print(f"Cropping from {current_len} → {new_len} timesteps...")

            # 🔥 NEW: overwrite file entirely if HDF5 is longer than CSV
            f.close()  # Close open handle
            os.remove(file_path)  # Truncate file (delete completely)

            # Recreate the HDF5 file with the cropped data
            with h5py.File(file_path, "w") as newf:
                newf.create_dataset(
                    dset_name,
                    data=dset[:new_len],
                    maxshape=(None, *dset.shape[1:]),
                    chunks=True,
                    dtype=dset.dtype,
                )
            print("✅ File truncated and recreated with cropped data.")
        else:
            print("✅ No cropping needed. Temporal lengths already match.")


def update_gridded_rainfall_cumulative(
    data_path=GR_RAINFALL_CUM_H5_PATH,
    temporal_data_path=GR_RAINFALL_CUM_TEMPORAL_PATH):
    """
    Combine newly downloaded gridded rainfall with existing data.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        temporal_data_path (str): Path to historic temporal data CSV.
    """

    try:
        # Crop historic data if historic spatial and temporal data are not the same size   
        if os.path.exists(GR_RAINFALL_CUM_H5_PATH) and os.path.exists(temporal_data_path):
            crop_historic_data(
                file_path=GR_RAINFALL_CUM_H5_PATH,
                temporal_data_path=temporal_data_path,
            )
            
        # Load new gridded rainfall data
        gridded_rainfall_new, gridded_rainfall_cumulative_last = load_new_gridded_rainfall_data()
        historic_dates = get_historic_dates()
        new_dates = get_new_dates()
        
        if gridded_rainfall_new is None:
            logging.info("No new files to process.")
            
        else:
            # Calculate cumulative values in the raw rainfall space used by the source H5.
            rainfall_3d_array_cumulative_last_unstandardised = gridded_rainfall_cumulative_last
    
    		# Cumulative sum most recent values
            new_cumsum = np.cumsum(np.concatenate((rainfall_3d_array_cumulative_last_unstandardised, gridded_rainfall_new), axis=0), axis=0)[1:]
    
            new_data = new_cumsum
            
            # Crop area to regions of interest
            regions_gdf = cleaning_utils.extract_regions()
            
            # Calculate total number of cells
            total_cells = new_data[0].shape[0] * new_data[0].shape[1]
            
            # Create new temporal data
            rainfall_cumulative_temporal = pd.DataFrame({'cumulative_rainfall': new_data.sum(axis=(1, 2))})
            rainfall_cumulative_temporal['date'] = new_dates
            
            # Loop through regions
            for i in range(len(regions_gdf)):
                region_data = regions_gdf.iloc[[i]]
                region_code = gridded_stats.region_to_code_dict[region_data['region'].values[0]]
                region_area = cleaning_utils.mask_regions(region_data, np.array(new_data))
                
                rainfall_cumulative_temporal[f"cumulative_rainfall_{region_code}"] = np.nansum(region_area, axis=(1, 2)) / (total_cells - np.sum(np.isnan(region_area[0])))
    
            # Append new data to HDF5
            with h5py.File(data_path, 'a') as hdf:
                dset = hdf.get('cumulative_rainfall')
                if dset is None:
                    dset = hdf.create_dataset(
                        'cumulative_rainfall',
                        shape=(0, new_data.shape[1], new_data.shape[2]),
                        maxshape=(None, new_data.shape[1], new_data.shape[2]),
                        chunks=(1, new_data.shape[1], new_data.shape[2]),
                        dtype=new_data.dtype,
                    )
                old_dataset_length = dset.shape[0]
                dset.resize(dset.shape[0] + new_data.shape[0], axis=0)
                dset[-new_data.shape[0]:] = new_data
                logging.info(f"Updated cumulative rainfall dataset shape: {dset.shape}")
            
            # Update temporal data
            rainfall_cumulative_temporal_historic = (
                pd.read_csv(temporal_data_path)[:old_dataset_length]
                if os.path.exists(temporal_data_path) else pd.DataFrame(columns=rainfall_cumulative_temporal.columns)
            ) # Crop to length of spatial data
            rainfall_cumulative_temporal_new = pd.concat([rainfall_cumulative_temporal_historic, rainfall_cumulative_temporal])
            
            # Save the updated temporal data
            rainfall_cumulative_temporal_new.to_csv(temporal_data_path, index=False)

    except Exception as e:
        logging.error(f"Error processing cumulative rainfall data: {e}")