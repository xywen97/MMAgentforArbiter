# import necessary package
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    logging.error("TensorFlow is not installed. Please install it to run the RNN model.")
    os._exit(1)  # Exit the script if TensorFlow is not available

# Load the dataset
wimbledon_data = pd.read_csv('Wimbledon_featured_matches.csv')

# Check the structure of the dataset
logging.info(f"Loaded data with shape: {wimbledon_data.shape}")
logging.info(f"Columns in the dataset: {wimbledon_data.columns.tolist()}")

# Preprocess the data: Filter and structure the data for time-series analysis
def preprocess_data(data):
    # Select relevant columns for analysis
    columns_of_interest = ['elapsed_time', 'p1_points_won', 'p2_points_won', 
                            'p1_unf_err', 'p2_unf_err', 'p1_ace', 'p2_ace', 
                            'point_victor', 'game_victor', 'set_victor']
    processed_data = data[columns_of_interest].copy()
    
    # Convert elapsed_time to seconds for easier manipulation
    processed_data['elapsed_time'] = pd.to_timedelta(processed_data['elapsed_time']).dt.total_seconds()
    
    # Create a time series index
    processed_data['time_index'] = np.arange(len(processed_data))
    
    logging.info("Data preprocessed for time-series analysis.")
    return processed_data

processed_data = preprocess_data(wimbledon_data)

# Save preprocessed data for later use
processed_data.to_csv('processed_wimbledon_data.csv', index=False)

# Implement ARIMA model
def fit_arima_model(series):
    # Fit ARIMA model
    model = sm.tsa.ARIMA(series, order=(1, 1, 1))  # Example order
    model_fit = model.fit()
    logging.info(f"ARIMA model fitted with summary: {model_fit.summary()}")
    return model_fit

# Fit the model on points won by player 1
arima_model = fit_arima_model(processed_data['p1_points_won'])

# Make predictions using the ARIMA model
def predict_arima(model, steps=10):
    forecast = model.forecast(steps)[0]
    logging.info(f"ARIMA forecast for next {steps} points: {forecast}")
    return forecast

arima_forecast = predict_arima(arima_model)

# Implement RNN using LSTM
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare data for RNN
def prepare_rnn_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Create input features and labels for RNN
X, y = prepare_rnn_data(processed_data['p1_points_won'].values, look_back=5)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the RNN model
rnn_model = create_rnn_model((X_train.shape[1], 1))
rnn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate RNN model
y_pred = rnn_model.predict(X_test)
logging.info(f"RNN model predictions: {y_pred.flatten()}")

# Logistic Regression for predicting momentum shifts
def prepare_logistic_data(data):
    # Creating features and labels for logistic regression
    features = data[['p1_points_won', 'p2_points_won', 'p1_unf_err', 'p2_unf_err']]
    labels = (data['game_victor'] != 0).astype(int)  # 1 if there's a momentum shift (game won)
    return features, labels

X_logistic, y_logistic = prepare_logistic_data(processed_data)

# Train Logistic Regression model
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)
logistic_model = LogisticRegression()
logistic_model.fit(X_train_logistic, y_train_logistic)

# Evaluate Logistic Regression model
y_logistic_pred = logistic_model.predict(X_test_logistic)
accuracy = accuracy_score(y_test_logistic, y_logistic_pred)
f1 = f1_score(y_test_logistic, y_logistic_pred)
logging.info(f"Logistic Regression Model Accuracy: {accuracy}, F1 Score: {f1}")

# Save the models and predictions for further analysis
rnn_model.save('rnn_model.h5')
# Note: ARIMA model cannot be saved directly with .save(), we will skip saving it for now
pd.DataFrame(y_pred).to_csv('rnn_predictions.csv', index=False)
pd.DataFrame({'actual': y_test_logistic, 'predicted': y_logistic_pred}).to_csv('logistic_predictions.csv', index=False)

# Final output logging
logging.info("Task 3 completed successfully. Models and predictions saved.")