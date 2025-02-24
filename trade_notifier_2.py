from forex_utils import fetch_forex_data
from lightGBM_forex_swing_trading_model import (
    generate_features,
    FEATURES as BUY_FEATURES,
)
from lightGBM_forex_swing_trading_sell import add_technical_indicators
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
    buy_history = pd.read_csv("model_logs/training_history.csv")
    sell_history = pd.read_csv("model_logs/sell_history.csv")

    message = ["Forex Trading Signals:\n"]

    for from_symbol, to_symbol in currency_pairs:
        pair = f"{from_symbol}_{to_symbol}"
        data = prepare_data_table(fetch_forex_data(from_symbol, to_symbol))
        data = generate_features(data)

        # Simplified model selection
        best_buy = (
            buy_history[buy_history.currency_pair == pair]
            .nlargest(1, "f1_score")
            .iloc[0]
        )
        best_sell = (
            sell_history[sell_history.currency_pair == pair].nlargest(1, "f1").iloc[0]
        )

        buy_model = joblib.load(best_buy["model_path"])
        sell_model = joblib.load(best_sell["model_path"])

        # Prediction simplified
        latest = data.iloc[-1]
        buy_pred = buy_model.predict([latest[BUY_FEATURES]])[0]
        sell_pred = int(sell_model.predict([latest[SELL_FEATURES]])[0] > 0.5)

        predict_and_plot(data.copy(), buy_model, sell_model, pair)

        message.append(
            f"{pair} Signal: {'BUY' if buy_pred else 'HOLD'} : {'SELL' if sell_pred else 'HOLD'}\n"
            f"Buy Model F1: {best_buy.f1_score:.2%}\n"
            f"Sell Model F1: {best_sell.f1:.2%}\n"
            f"Combined Accuracy: {(best_buy.accuracy + best_sell.accuracy) / 2:.2%}\n\n"
        )

    print("\n".join(message))


if __name__ == "__main__":
    main()
