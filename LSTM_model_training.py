import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import talib

# -----------------------------------
# Step 1: Data Preparation
# -----------------------------------


def load_data(file_path):
    """
    Load historical stock price data from a CSV file.
    Expected columns: date, open, high, low, close, volume
    """
    data = pd.read_csv(file_path)
    return data


def calculate_technical_indicators(data):
    """
    Calculate technical indicators using TA-Lib.
    Parameters:
    - data: DataFrame with columns ['open', 'high', 'low', 'close']
    Returns:
    - DataFrame with additional columns for technical indicators
    """
    # KD (Stochastic Oscillator)
    data["slowk"], data["slowd"] = talib.STOCH(
        data["high"],
        data["low"],
        data["close"],
        fastk_period=5,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )

    # RSI (Relative Strength Index)
    data["RSI"] = talib.RSI(data["close"], timeperiod=14)

    # BIAS (manual calculation)
    data["BIAS"] = (
        (data["close"] - data["close"].rolling(window=20).mean())
        / data["close"].rolling(window=20).mean()
        * 100
    )

    # Williams %R
    data["Williams% R"] = talib.WILLR(
        data["high"], data["low"], data["close"], timeperiod=14
    )

    # MACD
    data["macd"], data["macdsignal"], data["macdhist"] = talib.MACD(
        data["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )

    return data


def prepare_features(data):
    """
    Prepare the feature set by combining stock price data and technical indicators.
    """
    features = data[
        [
            "open",
            "high",
            "low",
            "close",
            "slowk",
            "slowd",
            "RSI",
            "BIAS",
            "Williams% R",
            "macd",
            "macdsignal",
            "macdhist",
        ]
    ]
    return features


def normalize_data(features):
    """
    Normalize the features using MinMaxScaler to ensure all values are on a similar scale.
    """
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


def create_sequences(data, sequence_length):
    """
    Create sequences of data points for the LSTM model to learn temporal patterns.
    Each sequence is of length 'sequence_length'.
    """
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i : i + sequence_length])
    return np.array(sequences)


def create_labels(data, sequence_length):
    """
    Create binary labels for buy signals.
    Label = 1 if the next day's close is higher than the current day's close (buy signal),
    Label = 0 otherwise.
    """
    labels = []
    for i in range(sequence_length, len(data)):
        labels.append(1 if data["close"][i] > data["close"][i - 1] else 0)
    return np.array(labels)


# -----------------------------------
# Step 2: Model Architecture
# -----------------------------------


def build_lstm_model(input_shape):
    """
    Build a multi-layer LSTM model with dropout layers to prevent overfitting.
    Input shape: (sequence_length, number_of_features)
    """
    model = Sequential()

    # First LSTM layer with return sequences
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Second LSTM layer with return sequences
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))

    # Third LSTM layer with return sequences
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))

    # Fourth LSTM layer (final)
    model.add(LSTM(50))
    model.add(Dropout(0.2))

    # Output layer for binary classification (buy signal or not)
    model.add(Dense(1, activation="sigmoid"))

    return model


# -----------------------------------
# Step 3: Training
# -----------------------------------


def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Compile and train the LSTM model using binary cross-entropy loss and Adam optimizer.
    """
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Use 20% of training data for validation
        verbose=1,
    )
    return history


# -----------------------------------
# Step 4: Evaluation
# -----------------------------------


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set using accuracy and loss metrics.
    Additional metrics like F1 score can be added as needed.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


# -----------------------------------
# Main Function to Run the Script
# -----------------------------------


def main():
    """
    Main function to load data, prepare features, build and train the model, and evaluate performance.
    """
    # Load and prepare data
    data = load_data("path_to_data.csv")  # Replace with your data file path
    data = calculate_technical_indicators(data)
    data = data.dropna()  # Add this line to handle NaN values
    features = prepare_features(data)
    scaled_features, scaler = normalize_data(features)

    # Define sequence length (e.g., 10 days of historical data)
    sequence_length = 10
    X = create_sequences(scaled_features, sequence_length)
    y = create_labels(data, sequence_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    history = train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
