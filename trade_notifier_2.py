from forex_utils import fetch_forex_data
from train_buy_5 import (
    generate_features,
    FEATURES as BUY_FEATURES,
)
from train_sell_3 import add_technical_indicators
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Simplified sell features list (removed redundant comments)
SELL_FEATURES = [
    "open",
    "high",
    "low",
    "close",  # Price features
    "RSI",
    "MACD",
    "MACD_signal",
    "ATR",
    "STOCH_K",
    "STOCH_D",
    "BB_upper",
    "BB_middle",
    "BB_lower",
    "momentum",
    "volatility",
    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_10",
]


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
    df = df[df.index.dayofweek < 5]

    # Add sell model features (must happen before generate_features)
    df = add_technical_indicators(df)

    return df


def predict_and_plot(data, buy_model, sell_model, pair):
    """Generate predictions and display interactive plot with signals"""
    data["buy_prediction"] = buy_model.predict(data[BUY_FEATURES])
    data["sell_prediction"] = (sell_model.predict(data[SELL_FEATURES]) > 0.5).astype(
        int
    )

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["close"], label="Closing Price", color="blue", alpha=0.5)

    # Combined signal plotting
    for signal_type, color, marker, label in [
        ("buy_prediction", "green", "^", "Buy Signal"),
        ("sell_prediction", "red", "v", "Sell Signal"),
    ]:
        signals = data[data[signal_type] == 1]
        plt.scatter(
            signals.index,
            signals["close"],
            color=color,
            marker=marker,
            s=100,
            label=label,
        )

    plt.title(f"{pair} Trading Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
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

    # Model loading simplified
    buy_history_5_day = pd.read_csv("model_logs/training_history.csv")
    sell_history_3_day = pd.read_csv("model_logs/sell_history.csv")

    message = ["Forex Trading Signals:\n"]

    for from_symbol, to_symbol in currency_pairs:
        pair = f"{from_symbol}_{to_symbol}"
        data = prepare_data_table(fetch_forex_data(from_symbol, to_symbol))
        data = generate_features(data)

        # Simplified model selection
        best_buy_5_day = (
            buy_history_5_day[buy_history_5_day.currency_pair == pair]
            .nlargest(1, "f1_score")
            .iloc[0]
        )
        best_sell_3_day = (
            sell_history_3_day[sell_history_3_day.currency_pair == pair]
            .nlargest(1, "f1")
            .iloc[0]
        )

        buy_model_5_day = joblib.load(best_buy_5_day["model_path"])
        sell_model_3_day = joblib.load(best_sell_3_day["model_path"])

        # Prediction simplified
        latest = data.iloc[-1]
        buy_pred_5_day = buy_model_5_day.predict([latest[BUY_FEATURES]])[0]
        sell_pred_3_day = int(
            sell_model_3_day.predict([latest[SELL_FEATURES]])[0] > 0.5
        )

        predict_and_plot(data.copy(), buy_model_5_day, sell_model_3_day, pair)

        message.append(
            f"{pair} Signal: {'BUY' if buy_pred_5_day else 'HOLD'} (5-day) : "
            f"{'SELL' if sell_pred_3_day else 'HOLD'} (3-day)\n"
            f"Buy Model (5-day) F1: {best_buy_5_day.f1_score:.2%}\n"
            f"Sell Model (3-day) F1: {best_sell_3_day.f1:.2%}\n"
            f"Combined Accuracy: {(best_buy_5_day.accuracy + best_sell_3_day.accuracy) / 2:.2%}\n\n"
        )

    print("\n".join(message))


if __name__ == "__main__":
    main()
