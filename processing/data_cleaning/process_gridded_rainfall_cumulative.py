# Import system libraries
import os

# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime

# Import compression libraries
import h5py

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_stats(stats_file_path, temporal=False):
    """
    Read the rainfall statistics file.

    Parameters:
        stats_file_path (str): Path to rainfall statistics file.
    """
    with open(stats_file_path, 'r') as f:
        lines = f.readlines()
    # Extract mean and std from the file
    rainfall_mean = float(lines[0 + temporal*2].split(': ')[1].strip())
    rainfall_std = float(lines[1 + temporal*2].split(': ')[1].strip())
    return rainfall_mean, rainfall_std


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
    
    
def get_historic_dates(data_path='data/historic/gridded_rainfall_cumulative_temporal.csv'):
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
        logging.error(f"File not found: {data_path}")
        return []
        
        
def get_new_dates(rainfall_data_path='data/historic/gridded_rainfall_temporal.csv',
                  cumulative_rainfall_data_path='data/historic/gridded_rainfall_cumulative_temporal.csv'):
    """
    Get list of new dates from pre-downloaded data.

    Parameters:
        data_path (str): Directory path of pre-downloaded temporal data.
    """
    try:
        gridded_rainfall_temporal = pd.read_csv(rainfall_data_path, index_col=0)
        gridded_rainfall_cumulative_temporal = pd.read_csv(cumulative_rainfall_data_path, index_col=0)
        new_dates = gridded_rainfall_temporal.iloc[len(gridded_rainfall_cumulative_temporal):].index.tolist()
        return new_dates
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        return []



def load_new_gridded_rainfall_data(temporal_data_path='data/historic/gridded_rainfall_temporal.csv',
								   cum_temporal_data_path='data/historic/gridded_rainfall_cumulative_temporal.csv', 
								   data_path='data/historic/gridded_rainfall.h5', 
								   cum_data_path='data/historic/gridded_rainfall_cumulative.h5'):
	"""
	Load new gridded rainfall data to be combined with cumulative sums.

	Parameters:
		temporal_data_path (str): Path to temporal gridded rainfall data.
		cum_temporal_data_path (str): Path to temporal gridded cumulative rainfall data.
		data_path (str): Path to gridded rainfall data.
		cum_data_path (str): Path to gridded cumulative rainfall data.
	"""

	# Identify the new recent historic gridded rainfall data
	len_old = len(pd.read_csv(cum_temporal_data_path))
	len_new = len(pd.read_csv(temporal_data_path))
	new_rainfall_indices = len_new - len_old

	# Open the new gridded rainfall data
	with h5py.File(data_path, 'r') as gridded_rainfall:
	    # Access the dataset (replace 'your_dataset' with the actual dataset name)
	    dataset = gridded_rainfall['rainfall']
	    
	    # Load the new rainfall grids
	    gridded_rainfall_new = dataset[-new_rainfall_indices:, :, :]

	# Open the last cumulative gridded rainfall data
	with h5py.File(cum_data_path, 'r') as gridded_rainfall_cumulative:
	    # Access the dataset (replace 'your_dataset' with the actual dataset name)
	    dataset = gridded_rainfall_cumulative['cumulative_rainfall']
	    
	    # Load the new rainfall grids
	    gridded_rainfall_cumulative_last = np.expand_dims(dataset[-1, :, :], axis=0)

	return gridded_rainfall_new, gridded_rainfall_cumulative_last


def update_gridded_rainfall_cumulative(
		data_path='data/historic/gridded_rainfall_cumulative.h5',
        temporal_data_path='data/historic/gridded_rainfall_cumulative_temporal.csv',
        stats_file_path='data/stats/gridded_rainfall_cumulative_stats.txt'):
    """
    Combine newly downloaded gridded rainfall with existing data.

    Parameters:
        download_path (str): Directory path to save downloaded TIF files.
        temporal_data_path (str): Path to historic temporal data CSV.
    """

    try:

        # Load new gridded rainfall data
        gridded_rainfall_new, gridded_rainfall_cumulative_last = load_new_gridded_rainfall_data()
        historic_dates = get_historic_dates()
        new_dates = get_new_dates()
        
        if len(gridded_rainfall_new) == 0:
            logging.info("No new files to process.")
            
        else:
            # Calculate cumulative values for new data using most recent past data array
            cum_rainfall_mean, cum_rainfall_std = read_stats(stats_file_path)
            rainfall_3d_array_cumulative_last_unstandardised = unstandardize_array(gridded_rainfall_cumulative_last, cum_rainfall_mean, cum_rainfall_std)
    
    		# Cumulative sum most recent values
            new_cumsum = np.cumsum(np.concatenate((rainfall_3d_array_cumulative_last_unstandardised, gridded_rainfall_new), axis=0), axis=0)[1:]
    
    		# Standard scale new data based on saved values
            new_data = standardize_array(new_cumsum, cum_rainfall_mean, cum_rainfall_std)
            
            # Update temporal data
            temporal_df = pd.DataFrame({'cumulative_rainfall': new_data.sum(axis=(1, 2))})
            temporal_df['date'] = new_dates
            temporal_mean, temporal_std = read_stats(stats_file_path, temporal=True)
            temporal_df['cumulative_rainfall'] = (temporal_df['cumulative_rainfall'] - temporal_mean) / temporal_std
            temporal_historic = pd.read_csv(temporal_data_path)
            temporal_updated = pd.concat([temporal_historic, temporal_df])
            temporal_updated.to_csv(temporal_data_path, index=False)
    
            # Append new data to HDF5
            with h5py.File(data_path, 'a') as hdf:
                dset = hdf['cumulative_rainfall']
                dset.resize(dset.shape[0] + new_data.shape[0], axis=0)
                dset[-new_data.shape[0]:] = new_data
                logging.info(f"Updated cumulative rainfall dataset shape: {dset.shape}")

    except Exception as e:
        logging.error(f"Error processing cumulative rainfall data: {e}")