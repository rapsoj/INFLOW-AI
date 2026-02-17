import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from scipy.stats import norm
from tqdm import tqdm

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

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
        X.append(predictors[i:i + look_back, :])
        y.append(target[i + look_back:i + look_back + predict_ahead, 0])
        indices.append(i)
    return np.array(X), np.array(y), indices

def transformer_encoder(inputs, positional_encoding, d_model, num_heads=8, ff_dim=128, dropout_rate=0.1):
    x = inputs + positional_encoding
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(d_model)(ff_output)
    return LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    sign_penalty = tf.reduce_mean(tf.where(tf.sign(y_true) != tf.sign(y_pred), 20.0, 1.0))
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_penalty = tf.reduce_mean(tf.square(sum_true - sum_pred))
    return mse * sign_penalty + 0.1 * sum_penalty

def monte_carlo_predictions(model, X, num_samples=100):
    all_preds = []
    @tf.function
    def predict_with_dropout(inputs):
        return model(inputs, training=True)
    for _ in range(num_samples):
        preds = predict_with_dropout(X)
        all_preds.append(preds.numpy())
    all_preds = np.array(all_preds)
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)

# -----------------------------
# TRAINING FUNCTIONS
# -----------------------------

def build_transformer_model(seq_len, num_features, predict_ahead, positional_encoding):
    input_layer = Input(shape=(seq_len, num_features))
    x = transformer_encoder(input_layer, positional_encoding, d_model=num_features)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(predict_ahead)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=custom_loss, metrics=['mae'])
    return model

def train_model(X_train, y_train, validation_split=0.1, epochs=500, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model = build_transformer_model(X_train.shape[1], X_train.shape[2], y_train.shape[1],
                                    positional_encoding=tf.convert_to_tensor(get_positional_encoding(X_train.shape[1], X_train.shape[2]), dtype=tf.float32))
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                        callbacks=[early_stopping])
    return model, history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MSE: {mse}, Test MAE: {mae}")
    return y_pred

def train_and_test(X, y, test_size=0.3, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    permutation = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[permutation], y_train[permutation]
    model, history = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    return model, history, X_train, X_test, y_train, y_test, y_pred

def train_full_and_save(X, y, model_path='temporal_model.keras'):
    model, _ = train_model(X, y)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model


if __name__ == "__main__":
    # Example: load your data
    temporal_data_seasonal_df = pd.read_csv('data/temporal_data_seasonal_df.csv', index_col=0).iloc[:804]
    predictors = temporal_data_seasonal_df.iloc[:, :-1].values
    target = temporal_data_seasonal_df.iloc[:, -1].values.reshape(-1, 1)
    
    X, y, indices = create_overlapping_sequences(predictors, target, look_back=36, predict_ahead=6)

    # Train on train/test split
    model, history, X_train, X_test, y_train, y_test, y_pred = train_and_test(X, y, test_size=0.3)

    # Train on full data and save
    full_model = train_full_and_save(X, y, model_path='temporal_model.keras')
