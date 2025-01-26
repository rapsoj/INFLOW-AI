# Load data manipulation libraries
import numpy as np
import pandas as pd

# Load machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Import statistical libraries
from scipy.stats import norm

# Import progress bar libraries
from tqdm import tqdm

# Set data for training
temporal_data_seasonal_df = pd.read_csv('data/temporal_data_seasonal_df.csv', index_col=0)
temporal_data_seasonal_df = temporal_data_seasonal_df.iloc[:804]

# Positional encoding function
def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    positional_encoding = np.zeros((seq_len, d_model))
    positional_encoding[:, 0::2] = np.sin(position * div_term)
    if d_model % 2 == 0:
        positional_encoding[:, 1::2] = np.cos(position * div_term)
    else:
        positional_encoding[:, 1::2] = np.cos(position * div_term[:-1])
    return positional_encoding

def create_overlapping_sequences(predictors, target, look_back=12, predict_ahead=6):
    X, y, indices = [], [], []
    for i in range(len(predictors) - look_back - predict_ahead + 1):
        X.append(predictors[i:i + look_back, :])  # Sequence of length look_back
        y.append(target[i + look_back:i + look_back + predict_ahead, 0])  # Predict next predict_ahead values
        indices.append(i)
    return np.array(X), np.array(y), indices

# Prepare sequences
look_back = 36
predict_ahead = 6
X, y, indices_seq = create_overlapping_sequences(temporal_data_seasonal_df.iloc[:,:-1].values, temporal_data_seasonal_df.iloc[:,-1].values.reshape(-1, 1), look_back=look_back, predict_ahead=predict_ahead)

# Split into training and testing sets
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices_seq, test_size=0.30, shuffle=False)

# Create a random permutation of indices
permutation = np.random.permutation(X_train.shape[0])

# Apply the permutation to both X_train and y_train
X_train = X_train[permutation]
y_train = y_train[permutation]

# If you also want to shuffle indices_train in the same way:
indices_train = np.array(indices_train)[permutation]

# Positional encoding dimensions
d_model = X_train.shape[2]
positional_encoding = positional_encoding = tf.convert_to_tensor(get_positional_encoding(X_train.shape[1], d_model), dtype=tf.float32)

def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout_rate=0.1):
    inputs += positional_encoding
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(d_model)(ff_output)
    return LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

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

# Model input shape
seq_len, num_features = X_train.shape[1], X_train.shape[2]

# Build the model
input_layer = Input(shape=(seq_len, num_features))
x = transformer_encoder(input_layer, d_model=d_model, num_heads=8, ff_dim=128, dropout_rate=0.1)
x = tf.keras.layers.Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.1)(x)
output_layer = Dense(predict_ahead)(x)

# Define the model
model_delta = Model(inputs=input_layer, outputs=output_layer)

# Custom loss function
def custom_loss(y_true, y_pred):

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

# Compile the model with the custom loss function
model_delta.compile(optimizer=Adam(learning_rate=0.0001), loss=custom_loss, metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
history = model_delta.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Predict the test set
y_pred = model_delta.predict(X_test)

# Evaluate the performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {np.mean(mse)}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {np.mean(mae)}')

# Intialise arrays to store unscaled values
y_test_unscaled = np.zeros(y_test.shape)
y_pred_unscaled = np.zeros(y_pred.shape)
inundation_test = np.zeros(y_test.shape)
inundation_pred = np.zeros(y_pred.shape)
lower_bounds = np.zeros(y_pred.shape)
upper_bounds = np.zeros(y_pred.shape)
lower_bounds_unscaled = np.zeros(y_pred.shape)
upper_bounds_unscaled = np.zeros(y_pred.shape)
lower_bound_unscaled_inundation = np.zeros(y_pred.shape)
upper_bound_unscaled_inundation = np.zeros(y_pred.shape)

# Unscale predicted indunation deltas and calculate true inundation
for i in tqdm(range(len(indices_test))):
  dates = temporal_data_seasonal_df.iloc[indices_test[i] + look_back: indices_test[i] + look_back + 6].index
  day_months = dates.str[5:]

  # Calculate prior inundation
  prior_date = temporal_data_seasonal_df.iloc[indices_test[i] + look_back - 1].name
  prior_inundation = inundation_temporal_unscaled.loc[prior_date].values[0]

  # Determine seasonal statistics to unscale predictions
  index_means = means.loc[day_months]['inundation_delta']
  index_stds = stds.loc[day_months]['inundation_delta']

  # Unscale predictions
  y_test_unscaled[i] = y_test[i] * index_stds.values + index_means.values
  y_pred_unscaled[i] = y_pred[i] * index_stds.values + index_means.values

  # Perform Monte Carlo sampling with drop out
  num_samples = 1000  # Number of MC dropout samples
  pred_data = X_test[i].copy()
  pred_data = np.expand_dims(pred_data, axis=0)
  preds_mean, preds_std = monte_carlo_predictions(model_delta, pred_data, num_samples=num_samples)
  preds_mean_unscaled = y_pred[i] * index_stds.values + index_means.values

  # Perform confidence correction
  confidence_correction = 3
  preds_std = preds_std * confidence_correction

  # Calculate confidence intervals using z-scores
  confidence_level = 0.95
  z_score = norm.ppf(1 - (1 - confidence_level) / 2)  # 1.96 for 95% CI
  lower_bounds[i] = preds_mean - z_score * preds_std
  upper_bounds[i] = preds_mean + z_score * preds_std

  # Unscale confidence intervals
  lower_bounds_unscaled[i] = lower_bounds[i] * index_stds.values + index_means.values
  upper_bounds_unscaled[i] = upper_bounds[i] * index_stds.values + index_means.values

  # Loop through predictions to calculate total inundation from deltas
  inundation_test[i] = prior_inundation + y_test_unscaled[i][0]
  inundation_pred[i] = prior_inundation + y_pred_unscaled[i][0]
  lower_bound_unscaled_inundation[i] = prior_inundation + lower_bounds_unscaled[i][0]
  upper_bound_unscaled_inundation[i] = prior_inundation + upper_bounds_unscaled[i][0]
  for j in range(1,6):
      inundation_test[i][j] = inundation_test[i][j - 1] + y_test_unscaled[i][j]
      inundation_pred[i][j] = inundation_pred[i][j - 1] + y_pred_unscaled[i][j]
      lower_bound_unscaled_inundation[i][j] = lower_bound_unscaled_inundation[i][j - 1] + lower_bounds_unscaled[i][j]
      upper_bound_unscaled_inundation[i][j] = upper_bound_unscaled_inundation[i][j - 1] + upper_bounds_unscaled[i][j]

# Evaluate the performance
print('')
mse = mean_squared_error(inundation_test, inundation_pred)
print(f'Mean Squared Error: {np.mean(mse)}')
mae = mean_absolute_error(inundation_test, inundation_pred)
print(f'Mean Absolute Error: {np.mean(mae)}')

# Provide context
std_ratio = inundation_temporal_unscaled['percent_inundation'].std() / mae
print(f'Standard Deviation Ratio: {std_ratio}x better than STD')

# Check calibration
calibration = (inundation_test > lower_bound_unscaled_inundation) & (inundation_test < upper_bound_unscaled_inundation)
print(f'Calibration by dekad: {sum(calibration) / len(calibration)}')

# Calculate interval size
interval_size = np.mean(upper_bound_unscaled_inundation - lower_bound_unscaled_inundation, axis=0)
interval_ratio = inundation_temporal_unscaled['percent_inundation'].std() / interval_size
print(f'Interval ratio: {interval_ratio}x better than STD')

# Save model
# model_delta.save('temporal_model.keras')
# model_delta.save('temporal_model_testing.keras')