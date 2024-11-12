import os
import numpy as np
import cupy as cp

# Constants for file paths
RAINFALL_ARRAY_PATH = 'outputs/rainfall_3d_array.npy'
CUMULATIVE_OUTPUT_PATH = 'outputs/rainfall_3d_array_cumulative.npy'
STATS_FILE_PATH = 'cumulative_rainfall_stats.txt'

# Load rainfall data (expected to be preprocessed in 'rainfall_gridded')
def load_rainfall_data(path):
    return np.load(path, allow_pickle=True)

# Function to calculate cumulative sum on GPU with chunking
def incremental_cumsum_gpu(arr, chunk_size=20):
    arr = cp.asarray(arr)  # Ensure the array is loaded as a CuPy array
    time, rows, cols = arr.shape
    cumulative_sum = cp.zeros((rows, cols), dtype=arr.dtype)  # Initialize cumulative sum
    cumulative_array = cp.zeros((chunk_size, rows, cols), dtype=arr.dtype)  # Temporary cumulative storage

    for t in range(0, time, chunk_size):
        end_t = min(t + chunk_size, time)
        temp_size = end_t - t

        # Update cumulative sum and store it in chunks
        for i in range(temp_size):
            valid_mask = ~cp.isnan(arr[t + i, :, :])  # Create a mask for valid entries
            cumulative_sum[valid_mask] += arr[t + i, valid_mask]  # Update cumulative sum for valid entries
            cumulative_array[i, :, :] = cumulative_sum  # Store in temporary array

        arr[t:t + temp_size, :, :] = cumulative_array[:temp_size, :, :]  # Update original array with cumulative sums

    return cp.asnumpy(arr)  # Convert back to NumPy array for further processing

# Calculate mean and standard deviation for cumulative rainfall
def calculate_statistics(array):
    time, rows, cols = array.shape
    total_sum = np.sum(array)
    total_sum_sq = np.sum(array ** 2)
    count = time * rows * cols

    mean = total_sum / count
    std_dev = np.sqrt(total_sum_sq / count - mean ** 2)

    # Save statistics to a text file
    with open(STATS_FILE_PATH, 'w') as f:
        f.write(f"Cumulative Rainfall Mean: {mean}\n")
        f.write(f"Cumulative Rainfall Std Dev: {std_dev}\n")

    return mean, std_dev

# Apply standard scaling to the cumulative rainfall array
def standardize_array(array, mean, std_dev):
    return (array - mean) / std_dev

# Main function to process cumulative rainfall data
def process_rainfall_gridded_cumulative(chunk_size=20):
    # Load original rainfall 3D array
    rainfall_3d_array = load_rainfall_data(RAINFALL_ARRAY_PATH)

    # Calculate cumulative sum over time
    rainfall_3d_array_cumulative = incremental_cumsum_gpu(rainfall_3d_array, chunk_size=chunk_size)

    # Calculate mean and standard deviation of cumulative rainfall
    cum_mean, cum_std = calculate_statistics(rainfall_3d_array_cumulative)

    # Standardize cumulative rainfall array
    rainfall_3d_array_cumulative = standardize_array(rainfall_3d_array_cumulative, cum_mean, cum_std)

    # Save cumulative rainfall array
    np.save(CUMULATIVE_OUTPUT_PATH, rainfall_3d_array_cumulative)
    print("Cumulative rainfall array saved to", CUMULATIVE_OUTPUT_PATH)

    return rainfall_3d_array_cumulative