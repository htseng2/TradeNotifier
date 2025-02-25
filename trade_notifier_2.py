from forex_utils import fetch_forex_data
from train_buy_5 import (
    generate_features,
    FEATURES as BUY_FEATURES,
)
from train_sell_3 import add_technical_indicators, FEATURES as SELL_FEATURES
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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


def send_email_notification(
    subject, message, sender_email, receiver_email, gmail_password
):
    """Send email notification using Gmail SMTP"""
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, gmail_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email notification: {e}")


def send_telegram_notification(message):
    """Send notification via Telegram bot"""
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Telegram notification sent successfully.")
        else:
            print(f"Failed to send Telegram notification: {response.text}")
    except Exception as e:
        print(f"Failed to send Telegram notification: {e}")


def send_notification(message):
    """Handle all notification methods"""
    # Email configuration
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    gmail_password = os.getenv("GMAIL_PASSWORD")
    email_subject = "Forex Trading Signals Alert"

    # Send both notifications
    send_telegram_notification(message)
    send_email_notification(
        email_subject, message, sender_email, receiver_email, gmail_password
    )


def main():
    currency_pairs = [
        ("USD", "TWD"),
        ("EUR", "TWD"),
        ("GBP", "TWD"),
        ("AUD", "TWD"),
        ("CHF", "TWD"),
        ("NZD", "TWD"),
        ("JPY", "TWD"),  # Not for investment, but track for fun
    ]

    # Load both model versions
    buy_history_5_day = pd.read_csv("model_logs/train_buy_5_history.csv")
    buy_history_20_day = pd.read_csv("model_logs/train_buy_20_history.csv")
    sell_history_3_day = pd.read_csv("model_logs/train_sell_3_history.csv")
    sell_history_20_day = pd.read_csv("model_logs/train_sell_20_history.csv")

    message = ["Forex Trading Signals:\n"]

    for from_symbol, to_symbol in currency_pairs:
        pair = f"{from_symbol}_{to_symbol}"
        data = prepare_data_table(fetch_forex_data(from_symbol, to_symbol))
        data = generate_features(data)

        # Get all model versions
        best_buy_5 = (
            buy_history_5_day[buy_history_5_day.currency_pair == pair]
            .nlargest(1, "f1")
            .iloc[0]
        )
        best_buy_20 = (
            buy_history_20_day[buy_history_20_day.currency_pair == pair]
            .nlargest(1, "f1")
            .iloc[0]
        )
        best_sell_3 = (
            sell_history_3_day[sell_history_3_day.currency_pair == pair]
            .nlargest(1, "f1")
            .iloc[0]
        )
        best_sell_20 = (
            sell_history_20_day[sell_history_20_day.currency_pair == pair]
            .nlargest(1, "f1")
            .iloc[0]
        )

        # Load all models
        buy_model_5 = joblib.load(best_buy_5["model_path"])
        buy_model_20 = joblib.load(best_buy_20["model_path"])
        sell_model_3 = joblib.load(best_sell_3["model_path"])
        sell_model_20 = joblib.load(best_sell_20["model_path"])

        # Generate predictions for all models
        latest = data.iloc[-1]
        buy_5_pred = buy_model_5.predict([latest[BUY_FEATURES]])[0]
        buy_20_pred = buy_model_20.predict([latest[BUY_FEATURES]])[0]
        sell_3_pred = int(sell_model_3.predict([latest[SELL_FEATURES]])[0] > 0.5)
        sell_20_pred = int(sell_model_20.predict([latest[SELL_FEATURES]])[0] > 0.5)

        # Generate plots for both model pairs
        # predict_and_plot(data.copy(), buy_model_5, sell_model_3, f"{pair} (5D/3D)")
        # predict_and_plot(data.copy(), buy_model_20, sell_model_20, f"{pair} (20D/20D)")

        # Update message with all signals
        message.append(
            f"{pair} Signals:\n"
            f"  Short-term: {'BUY' if buy_5_pred else 'HOLD'} (5-day) : {'SELL' if sell_3_pred else 'HOLD'} (3-day)\n"
            f"  Long-term:  {'BUY' if buy_20_pred else 'HOLD'} (20-day) : {'SELL' if sell_20_pred else 'HOLD'} (20-day)\n"
            f"  Latest Data: {latest.name.strftime('%Y-%m-%d')}\n"
            f"  Model Performance:\n"
            f"  - Buy 5-day F1: {best_buy_5.f1:.2%}\n"
            f"  - Buy 20-day F1: {best_buy_20.f1:.2%}\n"
            f"  - Sell 3-day F1: {best_sell_3.f1:.2%}\n"
            f"  - Sell 20-day F1: {best_sell_20.f1:.2%}\n\n"
        )

    send_notification("\n".join(message))


if __name__ == "__main__":
    main()
