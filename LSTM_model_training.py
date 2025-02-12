import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime


def create_sequences(data, lookback, forecast_horizon):
    """
    Create sequences for multi-step forecasting.

    Parameters:
        data (array-like): The 1D array of scaled values.
        lookback (int): Number of past time steps used as input.
        forecast_horizon (int): Number of future time steps to predict.

    Returns:
        X (np.array): Input sequences of shape (samples, lookback).
        y (np.array): Target sequences of shape (samples, forecast_horizon).
    """
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + forecast_horizon])
    return np.array(X), np.array(y)


def build_model(lookback, forecast_horizon):
    """
    Build and compile an LSTM model for multi-step forecasting.

    Parameters:
        lookback (int): Number of past time steps used as input.
        forecast_horizon (int): Number of future time steps to predict.

    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_horizon))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def plot_history(history):
    """
    Plot training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


def plot_all_forecasts(
    df_index, lookback, forecast_horizon, predictions, actuals, test_start_idx
):
    """
    Plot the forecast for all test samples on the same graph.

    For each test sample, the forecast corresponds to the date range:
    df.index[global_index + lookback : global_index + lookback + forecast_horizon]
    where global_index = sample index in the full sequence array.

    Parameters:
        df_index (pd.DatetimeIndex): The original datetime index of the data.
        lookback (int): Lookback period used for the sequences.
        forecast_horizon (int): Forecast horizon (number of steps predicted).
        predictions (np.array): Predicted values, shape (num_test_samples, forecast_horizon).
        actuals (np.array): Actual target values, shape (num_test_samples, forecast_horizon).
        test_start_idx (int): The starting index (in the full sequence array) for the test set.
    """
    plt.figure(figsize=(14, 8))

    num_test_samples = predictions.shape[0]
    for i in range(num_test_samples):
        # global index of the sample in the full sequence array
        global_idx = i + test_start_idx
        # Determine the date range for the forecast horizon
        # (The forecast starts at index global_idx + lookback)
        forecast_dates = df_index[
            global_idx + lookback : global_idx + lookback + forecast_horizon
        ]
        # Plot predicted forecast in blue and actual forecast in red with transparency.
        plt.plot(forecast_dates, predictions[i], color="blue", alpha=0.3)
        plt.plot(forecast_dates, actuals[i], color="red", alpha=0.3)

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Multi-Step Forecasts for All Test Samples")
    # Create custom legend handles
    custom_lines = [
        Line2D([0], [0], color="blue", lw=2, label="Predicted"),
        Line2D([0], [0], color="red", lw=2, label="Actual"),
    ]
    plt.legend(handles=custom_lines)
    plt.show()


def main():
    # ---------------------- Configuration ----------------------
    DATA_PATH = "Forex/Forex - USD.csv"  # Path to your CSV file
    LOOKBACK = 200  # Number of past time steps to consider
    FORECAST_HORIZON = 10  # Number of future prices to predict
    TEST_SPLIT = 0.2  # Fraction of data to reserve for testing
    EPOCHS = 50
    BATCH_SIZE = 32
    # -----------------------------------------------------------

    # Check if the data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file '{DATA_PATH}' not found.")
        return

    # Load and preprocess the data
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M:%S")
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    print("Data head:\n", df.head())

    # Scale the 'Close' prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["Close_scaled"] = scaler.fit_transform(df[["Close"]])
    print("Scaled data head:\n", df.head())

    # Create sequences
    data_values = df["Close_scaled"].values
    if len(data_values) < (LOOKBACK + FORECAST_HORIZON):
        print("Error: Not enough data for the specified lookback and forecast horizon.")
        return

    X, y = create_sequences(data_values, LOOKBACK, FORECAST_HORIZON)
    print("Shape of X (before reshaping):", X.shape)  # (samples, lookback)
    print("Shape of y:", y.shape)  # (samples, forecast_horizon)

    # Reshape X to add a feature dimension
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print("Shape of X (after reshaping):", X.shape)  # (samples, lookback, 1)

    # Split the data chronologically into training and testing sets
    split_index = int((1 - TEST_SPLIT) * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Build the LSTM model
    model = build_model(LOOKBACK, FORECAST_HORIZON)
    model.summary()

    # Ensure the directory exists
    os.makedirs("LSTM_models", exist_ok=True)

    # Create a timestamp string, e.g., "20231013_154530"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up callbacks with the model checkpoint saving to the specified directory and including the timestamp
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(
            f"LSTM_models/best_model_{timestamp}.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # Plot training history
    plot_history(history)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)

    # Make predictions on the test set
    predictions = model.predict(X_test)
    print("Predictions shape:", predictions.shape)  # (samples, forecast_horizon)

    # Invert scaling for predictions and actual values
    # Reshape to 2D array as expected by scaler.inverse_transform, then reshape back.
    predictions_flat = predictions.reshape(-1, 1)
    y_test_flat = y_test.reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions_flat).reshape(
        predictions.shape
    )
    y_test_original = scaler.inverse_transform(y_test_flat).reshape(y_test.shape)

    # Plot forecasts for all test samples in one graph using the original datetime index.
    # Note: The first sequence in X corresponds to data starting at df.index[0],
    # and each forecast target spans df.index[i+LOOKBACK : i+LOOKBACK+FORECAST_HORIZON].
    # For the test set, the global sequence index starts at `split_index`.
    plot_all_forecasts(
        df.index,
        LOOKBACK,
        FORECAST_HORIZON,
        predictions_original,
        y_test_original,
        test_start_idx=split_index,
    )


if __name__ == "__main__":
    main()
