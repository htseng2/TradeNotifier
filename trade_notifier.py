from cProfile import label
from buy_data_labeler_from_file import (
    add_atr,
    add_bbands,
    add_bearish_candlestick_patterns,
    add_daily_return,
    # add_label_column,
    add_macd,
    add_pivot_points,
    add_stoch,
    add_weekly_return,
    add_max_min,
    add_moving_averages,
    add_rsi,
    calculate_days_since_last_buy,
    calculate_return_since_last_buy,
)
from lightGBM_Alpha_buy_model_training import (
    add_technical_indicators,
    find_gross_expected_return,
    fetch_forex_data as fetch_forex_data_for_model_training,
)
from sell_data_labeler_from_file import add_max_min as add_max_min_sell
from data_labeler_from_file import add_label_column
from forex_utils import fetch_forex_data, prepare_data_table
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import requests
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()


def add_buy_sell_column(
    df,
    gross_expected_return=0.025,
    look_ahead_days=21,
):
    """Add buy and sell signal columns to the DataFrame."""
    df["buy"] = 0
    df["sell"] = 0

    # Changed range to cover full DataFrame length
    for index in range(len(df)):
        current_price = df["Close"].iloc[index]

        # Check for buy signal (only when there's enough future data)
        if index < len(df) - look_ahead_days:
            for future_index in range(index + 1, index + look_ahead_days + 1):
                expected_return = 1 + gross_expected_return
                threshold = current_price * expected_return
                if df["Close"].iloc[future_index] > threshold:
                    df.at[df.index[index], "buy"] = 1

        # Check for sell signal (for all valid indices)
        if index >= look_ahead_days:
            for past_index in range(index - look_ahead_days, index):
                expected_return = 1 + gross_expected_return
                threshold = df["Close"].iloc[past_index] * expected_return
                if df["Close"].iloc[index] > threshold:
                    df.at[df.index[index], "sell"] = 1

    return df


def check_indicator(df):
    # Extract the closing prices from the DataFrame
    closing_prices = df["Close"]

    # Calculate the RSI
    delta = closing_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Determine buy, sell, or hold based on RSI
    if rsi.iloc[-1] < 30:
        return "buy"
    elif rsi.iloc[-1] > 70:
        return "sell"
    else:
        return "hold"


def send_email_notification(
    subject, message, sender_email, receiver_email, gmail_password
):

    # Create the email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    # Send the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, gmail_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email notification: {e}")


def send_telegram_notification(message):
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
    # Email configuration
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    gmail_password = os.getenv("GMAIL_PASSWORD")
    email_subject = "Trade Notifier Alert"

    # Send notifications
    send_telegram_notification(message)
    send_email_notification(
        email_subject, message, sender_email, receiver_email, gmail_password
    )


def main():
    # Alpha Vantage API only supports up to 10 requests per minute, and 25 requests per day
    currency_pairs = [
        ("USD", "TWD"),
        ("EUR", "TWD"),
        ("GBP", "TWD"),
        ("AUD", "TWD"),
        ("CHF", "TWD"),
        ("NZD", "TWD"),
        ("JPY", "TWD"),  # Not for investment, but track for fun
    ]
    # Load model history and find best model for this currency pair
    model_history = pd.read_csv("lightGBM_model_history.csv")

    # Initialize an empty message
    full_message = ""

    for from_symbol, to_symbol in currency_pairs:
        data = fetch_forex_data(from_symbol, to_symbol)

        # Prepare the data table
        df = prepare_data_table(data)

        # Enhance the DataFrame with additional features
        df = add_max_min(df)
        df = add_max_min_sell(df)
        df = add_moving_averages(df)
        df = add_macd(df)
        df = add_atr(df)
        df = add_pivot_points(df)
        df = add_rsi(df)
        df = add_bbands(df)
        df = add_bearish_candlestick_patterns(df)
        df = add_technical_indicators(df)

        # Find the gross expected return for the model
        model_training_df = fetch_forex_data_for_model_training(
            f"Alpha_Vantage_Data/{from_symbol}_{to_symbol}.csv"
        )
        gross_expected_return = find_gross_expected_return(model_training_df)
        df = add_buy_sell_column(
            df,
            gross_expected_return,
        )

        # Plot the Close price and the buy/sell signal
        plt.figure(figsize=(12, 6))
        plt.plot(df["Close"], label="Close Price", alpha=0.7)

        df = calculate_return_since_last_buy(df)
        df = calculate_days_since_last_buy(df)

        # Filter models for current currency pair and sort by f1_1 score
        pair_models = model_history[
            (model_history["currency_pair(s)"] == from_symbol)
            & (model_history["model_filename"].notnull())
        ].sort_values("f1_1", ascending=False)

        if not pair_models.empty:
            best_model_path = pair_models.iloc[0]["model_filename"]
            gbm_buy = lgb.Booster(model_file=best_model_path)
            print(f"Loaded best model for {from_symbol}: {best_model_path}")
        else:
            # Fallback to default model if no history found
            gbm_buy = lgb.Booster(
                model_file=f"live_models/lightgbm_model_buy_signal.txt"
            )
            print(f"Using default model for {from_symbol}")

        # Select features using the model's expected feature names
        required_features = [col for col in gbm_buy.feature_name() if col in df.columns]
        missing_features = set(gbm_buy.feature_name()) - set(df.columns)

        if missing_features:
            print(f"Warning: Missing features {missing_features} in input data")

        df_buy = df[required_features]

        # Add the buy signal to the df_buy DataFrame
        df_buy["buy"] = (
            gbm_buy.predict(
                df_buy[required_features], num_iteration=gbm_buy.best_iteration
            )
            .round()
            .astype(int)
        )

        # Predict the label for the latest date
        y_pred_latest_buy = df_buy["buy"].iloc[-1]  # Now using stored prediction

        # Printe the head and tail of the DataFrame
        print(df_buy.head())
        print(df_buy.tail())
        print(df.head())
        print(df.tail())

        # Coverting the label to a string
        predicted_buy_str = ""
        if y_pred_latest_buy == 1:
            predicted_buy_str = "buy"
        else:
            predicted_buy_str = "hold"

        # Sell signals should be based on the historical buy signal
        df_buy["sell"] = 0
        look_ahead_days = 21
        for index in range(len(df_buy)):
            if index >= look_ahead_days:
                for past_index in range(index - look_ahead_days, index):
                    expected_return = 1 + gross_expected_return
                    threshold = df["Close"].iloc[past_index] * expected_return
                    if df_buy["buy"].iloc[past_index] == 1:
                        if df["Close"].iloc[index] > threshold:
                            df_buy.at[df_buy.index[index], "sell"] = 1

        # Coverting the label to a string
        predicted_sell_str = ""
        if df_buy["sell"].iloc[-1] == 1:
            predicted_sell_str = "sell"
        else:
            predicted_sell_str = "hold"

        # Append message for the current pair
        full_message += f"{from_symbol}/{to_symbol}: {predicted_buy_str}, {predicted_sell_str} ({gross_expected_return*100:.1f}%)\n\n"

    # Send the full notification
    send_notification(full_message)


if __name__ == "__main__":
    main()
