# Import cleaning utils
from processing import cleaning_utils

# Import data processing functions
from processing.data_cleaning import process_victoria
from processing.data_cleaning import process_albert
from processing.data_cleaning import process_kyoga
from processing.data_cleaning import process_rainfall
from processing.data_cleaning import process_teleconnections
from processing.data_cleaning import process_inundation
from processing.data_cleaning import process_gridded_rainfall
from processing.data_cleaning import process_gridded_rainfall_cumulative
from processing.data_cleaning import process_gridded_moisture

# Import data manipulation libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import statistical libraries
from scipy.stats import norm

# Import machine learning libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import data visualisation libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_future_dates(data):
	"""
	Get dates of next six dekads from most recent date in data.

	Parameters:
		data (df): Dataframe with updated temporal variables.
	"""
	last_date = data.index[-1]
	end_date = (datetime.today() + timedelta(days=60)).strftime("%Y-%m-%d")
	future_dates = cleaning_utils.get_dates_of_interest(last_date, end_date)[1:7]

	return future_dates


def check_if_new_data(data_path='data/temporal_data_seasonal_df.csv'):
	"""
	Prevents running data update if model is run during a dekad already in the data.

	Parameters:
		data_path: Path to temporal data from previous update.
	"""
	data = pd.read_csv(data_path, index_col=0)
	today_date = datetime.today().strftime("%Y-%m-%d")
	last_date = data.index[-1]
	possible_dates = cleaning_utils.get_dates_of_interest(last_date)
	missing_dates = cleaning_utils.get_dates_of_interest(last_date, today_date)[1:]
	if datetime.strptime(missing_dates[-1], "%Y-%m-%d") < datetime.today():
	  missing_dates = missing_dates[:-1]

	new_data = len(missing_dates) > 0

	return new_data


def update_data():
	"""
	Update all temporal data for model.
	"""
	if check_if_new_data():

		process_victoria.update_victoria()
		process_albert.update_albert()
		process_kyoga.update_kyoga()
		process_rainfall.update_rainfall()
		process_teleconnections.update_teleconnections()
		process_inundation.update_inundation()
		process_gridded_rainfall.update_gridded_rainfall()
		process_gridded_rainfall_cumulative.update_gridded_rainfall_cumulative()
		process_gridded_moisture.update_gridded_moisture()

		logging.info(f"Updated all temporal data.")

	else:
		logging.info(f"No new data to update.")


def create_dataframe():
	"""
	Create dataframe from recently refreshed data.
	"""
	# Load data
	victoria = pd.read_csv('data/historic/victoria.csv', index_col='date')
	albert = pd.read_csv('data/historic/albert.csv', index_col='date')
	kyoga = pd.read_csv('data/historic/kyoga.csv', index_col='date')
	rainfall = pd.read_csv('data/historic/rainfall.csv', index_col='date')
	teleconnections = pd.read_csv('data/historic/teleconnections.csv', index_col='date')
	inundation_temporal_scaled = pd.read_csv('data/historic/inundation_temporal_scaled.csv', index_col='date')
	gridded_rainfall_temporal = pd.read_csv('data/historic/gridded_rainfall_temporal.csv', index_col='date')
	gridded_rainfall_cumulative_temporal = pd.read_csv('data/historic/gridded_rainfall_cumulative_temporal.csv', index_col='date')
	gridded_moisture_temporal = pd.read_csv('data/historic/gridded_moisture_temporal.csv', index_col='date')

	# Calculate inundation delta
	inundation_temporal_unscaled = pd.read_csv('data/historic/inundation_temporal_unscaled.csv', index_col='date')
	inundation_temporal_delta = inundation_temporal_unscaled[['percent_inundation']].diff()
	inundation_temporal_delta.columns = ['inundation_delta']

	# Combine data into temporal dataframe
	temporal_data_df = pd.concat([
	    victoria,
	    albert,
	    kyoga,
	    rainfall,
	    teleconnections,
	    inundation_temporal_scaled.rename({'percent_inundation': 'inundation_temporal'}, axis=1),
	    gridded_rainfall_temporal.rename({'rainfall': 'rainfall_3d_temporal'}, axis=1),
	    gridded_rainfall_cumulative_temporal.rename({'cumulative_rainfall': 'rainfall_cumulative'}, axis=1),
	    gridded_moisture_temporal.rename({'moisture': 'moisture_3d_temporal'}, axis=1),
	    inundation_temporal_delta
	], axis=1).dropna()

	# Create month-day index and load saved seasonal statistics for scaling
	temporal_data_df['month_day'] = pd.to_datetime(temporal_data_df.index).strftime('%m-%d')
	temporal_data_seasonal_df = temporal_data_df.copy()
	means = pd.read_csv('data/stats/seasonal_means.csv', index_col='month_day')
	stds = pd.read_csv('data/stats/seasonal_stds.csv', index_col='month_day')

	# Normalize each variable using the corresponding means and stds
	for column in temporal_data_seasonal_df.columns[:-1]:  # Exclude the month_day column
	    # Create a new column for normalized values
	    normalized_values = []

	    for index, row in temporal_data_seasonal_df.iterrows():
	        month_day = row['month_day']

	        if month_day in means.index:
	            mean_value = means.loc[month_day, column]
	            std_value = stds.loc[month_day, column]
	            # Avoid division by zero
	            if std_value != 0:
	                normalized_value = (row[column] - mean_value) / std_value
	            else:
	                normalized_value = 0  # or np.nan, depending on your needs
	        else:
	            normalized_value = row[column]  # If no data for that month_day, keep original value

	        normalized_values.append(normalized_value)

	    # Assign the normalized values back to the DataFrame
	    temporal_data_seasonal_df[column] = normalized_values

	# Drop month-day column
	temporal_data_seasonal_df = temporal_data_seasonal_df.drop('month_day', axis=1)
	temporal_data_seasonal_df.to_csv('data/temporal_data_seasonal_df.csv')

	return temporal_data_seasonal_df


def custom_loss(y_true, y_pred):
    """
    Custom loss function for temporal model that adds a sign penalty to predictions.
    
    Parameters:
        y_true (array): True target values to predict.
        y_pred (array): Predicted target values from model.
    """
    # MSE for individual predictions
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    sign_penalty = tf.reduce_mean(tf.where(tf.sign(y_true) != tf.sign(y_pred), 20.0, 1.0))

    # Penalty for sum mismatch
    sum_true = tf.reduce_sum(y_true, axis=1)  # Sum of each sequence (across sequence steps)
    sum_pred = tf.reduce_sum(y_pred, axis=1)  # Sum of each predicted sequence
    sum_penalty = tf.reduce_mean(tf.square(sum_true - sum_pred))  # Penalise if sums are different

    # Combine the MSE and sum penalty
    total_loss = mse * sign_penalty + 0.1 * sum_penalty

    return total_loss


def predict_new_inundation(data, model_path='model/temporal_model.keras', custom_loss_function=custom_loss):
	"""
	Predict new inundation based on updated data.

	Parameters:
		data (df): Dataframe with fully updated temporal data.
		model_path (str): Path to pre-trained temporal model.
		custom_loss_function (function): Function to calculate custom loss for model.
	"""
	# Select data for prediction
	X_pred = create_dataframe().iloc[-36:,:-1].values
	X_pred_reshaped = np.expand_dims(X_pred, axis=0)

	# Load model and custom loss function
	model_delta = load_model(model_path, custom_objects={'custom_loss': custom_loss})
	y_pred = model_delta.predict(X_pred_reshaped)

	logging.info(f"New inundation predicted.")

	return y_pred, X_pred, model_delta


def monte_carlo_predictions(model, X, num_samples=100):
    """
    Generate predictions with Monte Carlo Dropout.
    Parameters:
        model: The trained model with dropout layers.
        X: Input data for prediction.
        num_samples: Number of Monte Carlo samples to draw.
    Returns:
        preds_mean: Mean of the predictions across samples.
        preds_std: Standard deviation of the predictions across samples.
    """

    # Store predictions for each sample
    all_preds = []

    # Define a callable for prediction with dropout enabled
    @tf.function
    def predict_with_dropout(inputs):
        return model(inputs, training=True)  # Enable dropout during prediction

    for _ in range(num_samples):
        preds = predict_with_dropout(X)  # Enable dropout during prediction
        all_preds.append(preds.numpy())  # Convert to numpy array

    # Convert to array for easy manipulation
    all_preds = np.array(all_preds)

    # Compute mean and standard deviation across samples
    preds_mean = np.mean(all_preds, axis=0)
    preds_std = np.std(all_preds, axis=0)

    return preds_mean, preds_std


def re_scale_predictions(data, y_pred, X_pred, future_dates, model_delta):
	"""
	Convert predictions back from seasonal deltas to unscaled inundation percentages.

	Parameters:
		data (df): Dataframe with updated temporal data.
		y_pred (array): Predicted seasonal inundation deltas for next six dekads.
		X_pred (array): Data from past 36 dekads.
		future_dates (list): List of dates for prediction in format 'YYYY-MM-DD'
		model_delta (model): Keras model trained on full temporal dataset.
	"""
	inundation_temporal_unscaled = pd.read_csv('data/historic/inundation_temporal_unscaled.csv', index_col='date')

	# Intialise arrays to store unscaled values
	y_pred_unscaled = np.zeros(y_pred.shape)
	inundation_pred = np.zeros(y_pred.shape)
	lower_bounds = np.zeros(y_pred.shape)
	upper_bounds = np.zeros(y_pred.shape)
	lower_bounds_unscaled = np.zeros(y_pred.shape)
	upper_bounds_unscaled = np.zeros(y_pred.shape)
	lower_bound_unscaled_inundation = np.zeros(y_pred.shape)
	upper_bound_unscaled_inundation = np.zeros(y_pred.shape)

	# Unscale predicted indunation deltas and calculate true inundation
	future_month_days = [date[5:] for date in future_dates]

	# Calculate prior inundation
	prior_date = data.iloc[-1].name
	prior_inundation = inundation_temporal_unscaled.loc[prior_date].values[0]

	# Determine seasonal statistics to unscale predictions
	index_means = means.loc[future_month_days]['inundation_delta']
	index_stds = stds.loc[future_month_days]['inundation_delta']

	# Unscale predictions
	y_test_unscaled[0] = y_test[0] * index_stds.values + index_means.values
	y_pred_unscaled[0] = y_pred[0] * index_stds.values + index_means.values

	# Perform Monte Carlo sampling with drop out
	num_samples = 1000  # Number of MC dropout samples
	preds_mean, preds_std = monte_carlo_predictions(model_delta, X_pred_reshaped, num_samples=num_samples)
	preds_mean_unscaled = y_pred[0] * index_stds.values + index_means.values

	# Perform confidence correction
	confidence_correction = 1
	preds_std = preds_std * confidence_correction

	# Calculate confidence intervals using z-scores
	confidence_level = 0.95
	z_score = norm.ppf(1 - (1 - confidence_level) / 2)  # 1.96 for 95% CI
	lower_bounds[0] = preds_mean - z_score * preds_std
	upper_bounds[0] = preds_mean + z_score * preds_std

	# Unscale confidence intervals
	lower_bounds_unscaled[0] = lower_bounds[0] * index_stds.values + index_means.values
	upper_bounds_unscaled[0] = upper_bounds[0] * index_stds.values + index_means.values

	# Loop through predictions to calculate total inundation from deltas
	inundation_pred[0] = prior_inundation + y_pred_unscaled[0][0]
	lower_bound_unscaled_inundation[0] = prior_inundation + lower_bounds_unscaled[0][0]
	upper_bound_unscaled_inundation[0] = prior_inundation + upper_bounds_unscaled[0][0]
	for j in range(1,6):
	    inundation_pred[0][j] = inundation_pred[0][j - 1] + y_pred_unscaled[0][j]
	    lower_bound_unscaled_inundation[0][j] = lower_bound_unscaled_inundation[0][j - 1] + lower_bounds_unscaled[0][j]
	    upper_bound_unscaled_inundation[0][j] = upper_bound_unscaled_inundation[0][j - 1] + upper_bounds_unscaled[0][j]


	return inundation_pred, lower_bound_unscaled_inundation, upper_bound_unscaled_inundation


def export_csv(inundation_pred, lower_bound_unscaled_inundation, upper_bound_unscaled_inundation, future_dates):
	"""
	Export CSV with predictions and 95% upper and lower confidence intervals.
	"""
	predictions = pd.DataFrame({'lower_bound_95': lower_bound_unscaled_inundation[0],
	                            'percent_inundation_prediction': inundation_pred[0],
	                            'upper_bound_95': upper_bound_unscaled_inundation[0]}, index=future_dates)

	predictions.to_csv(f'predictions/inundation_predictions_{predictions.index[0]}_to_{predictions.index[-1]}.csv')


def export_graphs(data, future_dates, inundation_pred, lower_bound_unscaled_inundation,
			      upper_bound_unscaled_inundation, inundation_temporal_unscaled_path='data/historic/inundation_temporal_unscaled.csv'):
	"""
	Export various graphs plotting the new inundation predictions.
	"""
	# Convert dates to datetime format for better control over x-axis formattingdates.index = pd.to_datetime(dates.index)  # Ensure dates are in datetime format
	pred_dates = pd.to_datetime(future_dates)    # Ensure pred_dates are in datetime format
	pred_dates = np.insert(pred_dates, 0, temporal_data_seasonal_df.index[-1])

	# Format prediction data
	inundation_pred_formatted = np.insert(inundation_pred[0], 0, inundation_temporal_unscaled['percent_inundation'].iloc[-1])
	lower_bound_unscaled_inundation_formatted = np.insert(lower_bound_unscaled_inundation[0], 0, inundation_temporal_unscaled['percent_inundation'].iloc[-1])
	upper_bound_unscaled_inundation_formatted =  np.insert(upper_bound_unscaled_inundation[0], 0, inundation_temporal_unscaled['percent_inundation'].iloc[-1])

	# Create three different plots at different scales
	x_lims = [data.index[0], data.index[-180], data.index[-36]]
	x_lim_names = ['total_record', 'past_five_years', 'past_year']
	for x_lim in range(len(x_lims)):

		# Plot the test sequence true vs predicted values
		plt.figure(figsize=(10, 6))
		plt.plot(pd.to_datetime(temporal_data_seasonal_df.index), inundation_temporal_unscaled['percent_inundation'].iloc[1:], label='Flood Coverage per MODIS Satellite Data', marker='o', linestyle='-', color='blue')
		plt.plot(pred_dates, inundation_pred_formatted, label='Predicted Flood Coverage (Updated November 1st)', marker='x', linestyle='--', color='red')

		# Fill the area between the lower and upper bounds
		plt.fill_between(pred_dates,
		                 lower_bound_unscaled_inundation_formatted,  # Lower bound
		                 upper_bound_unscaled_inundation_formatted,  # Upper bound
		                 color='red', alpha=0.2, label='95% Confidence Interval')

		# Format the x-axis to show only the year
		plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
		plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set ticks to appear each year

		plt.xlabel('Year')
		plt.ylabel('Flood Coverage Over INFLOW Study Area (%)')
		plt.ylim(inundation_temporal_unscaled['percent_inundation'].min() - 0.01, inundation_temporal_unscaled['percent_inundation'].max() + 0.01)
		plt.title(f"Predicted Flood Coverage Over INFLOW Study Area, {pred_dates[1].date().strftime('%Y-%m-%d')} to {pred_dates.max().date().strftime('%Y-%m-%d')}")
		plt.xlim(pd.Timestamp(x_lim), pd.Timestamp(future_dates[-1]) + timedelta(days=100))
		plt.legend()
		plt.grid(True)
		plt.xticks(rotation=45)

		plt.savefig(f"predictions/graphs/prediction_{pred_dates[1].date().strftime('%Y-%m-%d')}_to_{pred_dates.max().date().strftime('%Y-%m-%d')}_{x_lim_names[x_lim]}.png", dpi=300)
		plt.close()

	# Convert the index to a datetime index if it's not already
	inundation_df = pd.DataFrame(inundation_temporal_unscaled, columns=['percent_inundation'])
	inundation_df.index = pd.to_datetime(inundation_temporal_unscaled.index)
	inundation_df['predicted'] = False

	# Create dataframe for predicted inundation
	predicted_inundation_df = pd.DataFrame(inundation_pred_formatted, columns=['percent_inundation'])
	predicted_inundation_df.index = pd.to_datetime(pred_dates)
	predicted_inundation_df['predicted'] = True

	# Combine inundation
	combined_inundation = pd.concat([inundation_df, predicted_inundation_df])

	# Extract the year and the day of year from the date index
	combined_inundation['Year'] = combined_inundation.index.year
	combined_inundation['DayOfYear'] = combined_inundation.index.dayofyear

	# Create a colormap for other years (shades of blue and green)
	colors = list(mcolors.TABLEAU_COLORS.values())  # Colors from Tableau color palette
	cmap = plt.get_cmap('winter')  # Blue-green colormap

	# Plot each year's data on the same plot
	fig, ax = plt.subplots(figsize=(10, 6))

	# Loop through each year and plot the percent inundation
	for i, (year, group) in enumerate(combined_inundation.groupby('Year')):
	    if year == datetime.today().year:
	        # Separate the last 7 predictions from the 2024 data
	        final_predictions = group[group['predicted'] == True]

	        # Plot the current year data up to the last 7 points
	        ax.plot(group['DayOfYear'][:-len(final_predictions)], group['percent_inundation'][:-len(final_predictions)], color='red', label=year, linewidth=2)

	        # Plot the last 7 points as dashed
	        ax.plot(final_predictions['DayOfYear'], final_predictions['percent_inundation'], color='red', linestyle='--', linewidth=2)

	        # Fill the area between the lower and upper bounds for the last 7 predictions
	        last_seven_dates = final_predictions.index  # Use the date index for the last 7 points
	        ax.fill_between(final_predictions['DayOfYear'],
	                        lower_bound_unscaled_inundation_formatted[-len(final_predictions):],  # Lower bound of last 7 predictions
	                        upper_bound_unscaled_inundation_formatted[-len(final_predictions):],  # Upper bound of last 7 predictions
	                        color='red', alpha=0.2, label='95% Confidence Interval')
	    else:
	        # Assign different shades of blue and green for other years
	        color = cmap(i / len(combined_inundation['Year'].unique()))  # Normalize index for colormap
	        ax.plot(group['DayOfYear'], group['percent_inundation'], color=color, label=year, alpha=0.6)

	    # Add year label at the end of each line
	    ax.text(group['DayOfYear'].iloc[-1], group['percent_inundation'].iloc[-1],
	            str(year), color='black' if year == datetime.today().year else color,
	            fontsize=10, verticalalignment='center')

	# Formatting the plot
	ax.set_title(f"Flood Coverage Over INFLOW Study Area, {pred_dates[1].date().strftime('%Y-%m-%d')} to {pred_dates.max().date().strftime('%Y-%m-%d')}, ({datetime.today().year} Highlighted in Red)")
	ax.set_xlabel('Month of Year')
	ax.set_ylabel('Flood Coverage Over INFLOW Study Area (%)')
	ax.xaxis.set_major_locator(mdates.MonthLocator())  # Optional: Major ticks by month
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Format x-axis labels as months

	# Remove the legend (year labels are now on the lines)
	ax.legend().remove()

	plt.tight_layout()
	plt.savefig(f"predictions/graphs/prediction_{pred_dates[1].date().strftime('%Y-%m-%d')}_to_{pred_dates.max().date().strftime('%Y-%m-%d')}_year_by_year_comparison.png", dpi=300)
	plt.close()


def main():
	
	"""
	Main function to run model produce updated predictions.
	"""
	
	try: 
	    update_data()
	    future_dates = get_future_dates()
	    data = create_dataframe()
	    y_pred, X_pred, model_delta = predict_new_inundation(data)
	    inundation_pred, lb_pred, up_pred = re_scale_predictions(data, y_pred, X_pred, future_dates, model_delta)
	    export_csv(inundation_pred, lb_pred, up_pred, future_dates)
	    export_graphs(data, future_dates, inundation_pred, lb_pred, up_pred)

	    logging.info(f"Predictions exported.")

	except Exception as e:
	    print(f"Error occurred while exporting predictions: {e}")


if __name__ == "__main__":
	main()
