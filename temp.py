import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl

# Function to fill missing values by averaging the previous and next columns
def fill_missing_with_average(df):
    for col in df.columns:
        if df[col].isna().any():
            for i in range(1, len(df) - 1):
                if pd.isna(df[col].iloc[i]):
                    prev_col = df.columns[df.columns.get_loc(col) - 1]  # Previous column
                    next_col = df.columns[df.columns.get_loc(col) + 1]  # Next column
                    # Use .loc to ensure you're working with a copy
                    df.loc[i, col] = (df.loc[i, prev_col] + df.loc[i, next_col]) / 2
    return df

# Function to create data for RNN
def create_rnn_data(df, features, target, window_size=3):
    data_X = []
    data_y = []
    
    # Iterate over the rows and create sliding window of size 3
    for i in range(window_size, len(df)):
        X = df[features].iloc[i - window_size:i].values  # Last 3 days' features
        y = df[target].iloc[i]  # AQI of the current day
        data_X.append(X)
        data_y.append(y)
    
    return np.array(data_X), np.array(data_y)

def test_window_size(window_size, results):
    df = pd.read_csv("combined_data.csv")
    df = fill_missing_with_average(df)
    df_filled = df.copy()

    for col in df_filled.select_dtypes(include=['float64', 'int64']).columns:
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        
    features = ['vis', 'tmp', 'dew', 'slp', 'wnd_direc', 'wnd_scale', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    target = 'AQI'

    # Prepare the data (features for the last 3 days to predict AQI for the current day)
    X, y = create_rnn_data(df_filled, features, target, window_size=window_size)
    
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    log_dir = "logs/fit"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = Sequential()
    model.add(SimpleRNN(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=1e-4, clipvalue=1.0)  # Clip gradients to a maximum value
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Train the model
    history = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_test, y_test), 
        verbose=1,
        callbacks=[tensorboard_callback]  # Add TensorBoard callback here
    )

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')
    results.append({'window_size': window_size, 'test_loss': test_loss, 'test_mae': test_mae})
    
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred.flatten()
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)

    # Calculate control limits
    upper_control_limit = mean_residuals + 3 * std_residuals
    lower_control_limit = mean_residuals - 3 * std_residuals
    
    df['day'] = pd.to_datetime(df['day'], errors='coerce')
    dates = df['day'][-len(y_pred):] 

    # Plot the residuals and control chart
    plt.figure(figsize=(20, 6))
    plt.plot(dates, residuals, label='Residuals')
    plt.axhline(y=mean_residuals, color='green', linestyle='--', label='Mean Residual')
    plt.axhline(y=upper_control_limit, color='red', linestyle='--', label='Upper Control Limit (UCL)')
    plt.axhline(y=lower_control_limit, color='red', linestyle='--', label='Lower Control Limit (LCL)')
    plt.title('Residuals and Control Chart')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Step 6: Set x-ticks to only show the beginning of each year
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Place a tick at the beginning of each year
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.legend(loc='upper right')
    plt.savefig(f"results/residuals_window_{window_size}.png")
    plt.show()
    
    # Print the control limits for reference
    print(f'Mean Residuals: {mean_residuals}')
    print(f'Standard Deviation of Residuals: {std_residuals}')
    print(f'Upper Control Limit (UCL): {upper_control_limit}')
    print(f'Lower Control Limit (LCL): {lower_control_limit}')
    
results = []

window_sizes = [3]
for window_size in window_sizes:
    test_window_size(window_size, results)

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("results/window_size_results.csv", index=False)

# Plot the relationship between window_size and MAE (or Loss)
plt.figure(figsize=(10, 6))
plt.plot(results_df['window_size'], results_df['test_mae'], label='Test MAE', marker='o')
plt.xlabel('Window Size')
plt.ylabel('Test MAE')
plt.title('Window Size vs Test MAE')
plt.grid(True)
plt.savefig("results/window_size_vs_mae.png")  # Save this plot as PNG
plt.show()