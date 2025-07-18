# Import system libraries
import os

# Import cleaning utils
from .. import cleaning_utils

# Import statistics
from data.stats import gridded_data_stats

# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime

# Import compression libraries
import h5py

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_stats(region='all'):
    """
    Read the gridded data statistics file.
    """
    rainfall_cumulative_mean = gridded_data_stats.gridded_rainfall_cumulative_stats[region]['mean']
    rainfall_cumulative_std = gridded_data_stats.gridded_rainfall_cumulative_stats[region]['std']
    
    return rainfall_cumulative_mean, rainfall_cumulative_std


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
	if new_rainfall_indices <= 0:
	    return None, None
    
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
    temporal_data_path='data/historic/gridded_rainfall_cumulative_temporal.csv'):
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
        
        if gridded_rainfall_new is None:
            logging.info("No new files to process.")
            
        else:
            # Calculate cumulative values for new data using most recent past data array
            cum_rainfall_mean, cum_rainfall_std = read_stats()
            rainfall_3d_array_cumulative_last_unstandardised = unstandardize_array(gridded_rainfall_cumulative_last, cum_rainfall_mean, cum_rainfall_std)
    
    		# Cumulative sum most recent values
            new_cumsum = np.cumsum(np.concatenate((rainfall_3d_array_cumulative_last_unstandardised, gridded_rainfall_new), axis=0), axis=0)[1:]
    
    		# Standard scale new data based on saved values
            new_data = standardize_array(new_cumsum, cum_rainfall_mean, cum_rainfall_std)
            
            # Crop area to regions of interest
            regions_gdf = cleaning_utils.extract_regions()
            
            # Calculate total number of cells
            total_cells = new_data[0].shape[0] * new_data[0].shape[1]
            
            # Create new temporal data
            rainfall_cumulative_temporal = pd.DataFrame({'cumulative_rainfall': new_data.sum(axis=(1, 2))})
            rainfall_cumulative_temporal['date'] = new_dates
            temporal_mean, temporal_std = read_stats(region='all_temporal')
            rainfall_cumulative_temporal['cumulative_rainfall'] = (rainfall_cumulative_temporal['cumulative_rainfall'] - temporal_mean) / temporal_std
            
            # Loop through regions
            for i in range(len(regions_gdf)):
                region_data = regions_gdf.iloc[[i]]
                region_code = gridded_data_stats.region_to_code_dict[region_data['region'].values[0]]
                region_area = cleaning_utils.mask_regions(region_data, np.array(new_data))
                
                # Get stats for region
                temporal_mean_region, temporal_std_region = read_stats(region=region_code)
                rainfall_cumulative_temporal[f"cumulative_rainfall_{region_code}"] = np.nansum(region_area, axis=(1, 2)) / (total_cells - np.sum(np.isnan(region_area[0])))
                rainfall_cumulative_temporal[f"cumulative_rainfall_{region_code}"] = (rainfall_cumulative_temporal[f"cumulative_rainfall_{region_code}"] - temporal_mean_region) / temporal_std_region
    
            # Append new data to HDF5
            with h5py.File(data_path, 'a') as hdf:
                dset = hdf['cumulative_rainfall']
                dset.resize(dset.shape[0] + new_data.shape[0], axis=0)
                dset[-new_data.shape[0]:] = new_data
                logging.info(f"Updated cumulative rainfall dataset shape: {dset.shape}")
                new_dataset_length = dset.shape[0]
            
            # Update temporal data
            rainfall_cumulative_temporal_historic = pd.read_csv(temporal_data_path)[:new_dataset_length] # Crop to length of spatial data
            rainfall_cumulative_temporal_new = pd.concat([rainfall_cumulative_temporal_historic, rainfall_cumulative_temporal])
            
            # Save the updated temporal data
            rainfall_cumulative_temporal_new.to_csv(temporal_data_path, index=False)

    except Exception as e:
        logging.error(f"Error processing cumulative rainfall data: {e}")