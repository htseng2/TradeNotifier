from cProfile import label
from buy_data_labeler_from_file import (
    add_atr,
    add_bbands,
    add_bearish_candlestick_patterns,
    add_buy_sell_column,
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
from lightGBM_Alpha_buy_model_training import add_technical_indicators
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

# Load environment variables from .env file
load_dotenv()


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
        ("SGD", "TWD"),
        ("GBP", "TWD"),
        ("AUD", "TWD"),
        ("CHF", "TWD"),
        ("CAD", "TWD"),
        ("JPY", "TWD"),
        ("HKD", "TWD"),
        ("NZD", "TWD"),
    ]
    # Load model history and find best model for this currency pair
    model_history = pd.read_csv("lightGBM_model_history.csv")

    # Initialize an empty message
    full_message = ""

    # Parameters for the label column
    annual_expected_return = 0.20
    spread = 0.02  # Spread is transaction cost usually from 0.005 to 0.03
    holding_period = (1, 60)
    look_ahead_days = 21
    expected_return_per_trade = 0.005

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
        # df = add_label_column(
        #     df,
        #     annual_expected_return,
        #     holding_period,
        #     spread,
        #     look_ahead_days,
        #     expected_return_per_trade,
        # )
        df = add_technical_indicators(df)
        df = add_buy_sell_column(
            df,
            annual_expected_return,
            holding_period,
            spread,
            look_ahead_days,
            expected_return_per_trade,
        )
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

        gbm_sell = lgb.Booster(model_file=f"live_models/lightgbm_model_sell_signal.txt")
        # Select features for sell model
        sell_required_features = [
            col for col in gbm_sell.feature_name() if col in df.columns
        ]
        sell_missing = set(gbm_sell.feature_name()) - set(df.columns)

        if sell_missing:
            print(f"Sell model missing features: {sell_missing}")

        df_sell = df[sell_required_features]

        # Printe the head and tail of the DataFrame
        print(df_buy.head())
        print(df_buy.tail())
        print(df_sell.head())
        print(df_sell.tail())

        # Predict the label for the latest date
        X_latest_buy = df_buy.iloc[-1:]  # Get the latest row
        y_pred_latest_buy = gbm_buy.predict(
            X_latest_buy, num_iteration=gbm_buy.best_iteration
        )
        X_latest_sell = df_sell.iloc[-1:]  # Get the latest row
        y_pred_latest_sell = gbm_sell.predict(
            X_latest_sell, num_iteration=gbm_sell.best_iteration
        )
        # Print the most recent date in the dataset
        print(
            f"Latest analyzed data point for {from_symbol}/{to_symbol}: {df.index[-1].strftime('%Y-%m-%d')}"
        )
        # Convert probabilities to class labels
        buy = 1 if y_pred_latest_buy[0] >= 0.5 else 0
        sell = 1 if y_pred_latest_sell[0] >= 0.5 else 0
        # Coverting the label to a string
        predicted_buy_str = ""
        if buy == 1:
            predicted_buy_str = "buy"
        else:
            predicted_buy_str = "hold"

        predicted_sell_str = ""
        if sell == 1:
            predicted_sell_str = "sell"
        else:
            predicted_sell_str = "hold"

        # Check the indicator
        # indicator = check_indicator(df)
        # print(f"Indicator for {from_symbol}/{to_symbol}: ", indicator)

        # Append message for the current pair
        full_message += (
            # f"The current RSI indicator for {from_symbol}/{to_symbol} is: {indicator}\n"
            f"{from_symbol}/{to_symbol}: {predicted_buy_str}, {predicted_sell_str}\n\n"
        )

    # Send the full notification
    send_notification(full_message)


if __name__ == "__main__":
    main()
