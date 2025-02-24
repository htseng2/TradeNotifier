from forex_utils import fetch_forex_data
from lightGBM_forex_swing_trading_model import (
    generate_features,
    generate_labels,
    objective,
    train_final_model,
    FEATURES,
)
import optuna
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os


def prepare_data_table(data):
    """Convert API response to cleaned DataFrame"""
    daily_data = data.get("Time Series FX (Daily)", {})
    df = pd.DataFrame.from_dict(daily_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert and rename columns
    df = df.astype(float)
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
        }
    )[["open", "high", "low", "close"]]

    # Filter out weekends
    return df[df.index.dayofweek < 5]


def predict_and_plot(data, model, pair):
    """Generate predictions and display interactive plot"""
    # Make predictions for entire dataset
    data["prediction"] = model.predict(data[FEATURES])

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot closing price
    plt.plot(data.index, data["close"], label="Closing Price", color="blue", alpha=0.5)

    # Plot buy signals
    buy_signals = data[data["prediction"] == 1]
    plt.scatter(
        buy_signals.index,
        buy_signals["close"],
        color="green",
        marker="^",
        s=100,
        label="Buy Signal",
    )

    # Format plot
    plt.title(f"{pair} Trading Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Show interactive plot
    plt.show()


def main():
    currency_pairs = [
        ("USD", "TWD"),
        # ("EUR", "TWD"),
        # ("GBP", "TWD"),
        # ("AUD", "TWD"),
        # ("CHF", "TWD"),
        # ("NZD", "TWD"),
        # ("JPY", "TWD"),  # Not for investment, but track for fun
    ]

    # Load model training history
    history_df = pd.read_csv("model_logs/training_history.csv")

    full_message = "Forex Trading Signals:\n\n"

    for from_symbol, to_symbol in currency_pairs:
        pair = f"{from_symbol}_{to_symbol}"
        data = fetch_forex_data(from_symbol, to_symbol)
        data = prepare_data_table(data)
        # Generate features and labels
        data = generate_features(data)

        # Find best model for this currency pair
        best_model = (
            history_df[history_df["currency_pair"] == pair]
            .sort_values("f1_score", ascending=False)
            .iloc[0]
        )

        # Load model and make predictions
        model = joblib.load(best_model["model_path"])
        latest_prediction = model.predict(
            data[FEATURES].iloc[-1].values.reshape(1, -1)
        )[0]

        # Generate predictions and plot
        predict_and_plot(data.copy(), model, pair)

        # Generate signal message
        signal = "BUY" if latest_prediction == 1 else "HOLD"
        full_message += (
            f"{pair} Signal: {signal}\n"
            f"Model F1 Score: {best_model['f1_score']:.2%}\n"
            f"Accuracy: {best_model['accuracy']:.2%}\n\n"
        )

    print(full_message)


if __name__ == "__main__":
    main()
