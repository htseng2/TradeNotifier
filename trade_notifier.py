from cProfile import label
from data_labeler_from_file import (
    add_atr,
    add_bbands,
    add_bearish_candlestick_patterns,
    add_daily_return,
    add_label_column,
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
from forex_utils import fetch_forex_data, prepare_data_table
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import requests
import lightgbm as lgb

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
        ("CNY", "TWD"),
        ("EUR", "TWD"),
        ("NZD", "TWD"),
        ("SGD", "TWD"),
        ("GBP", "TWD"),
        ("AUD", "TWD"),
        ("CHF", "TWD"),
        ("CAD", "TWD"),
        ("JPY", "TWD"),
        # Less financially sound and less liquid currencies
        # ("HKD", "TWD"),
        # ("DKK", "TWD"),
    ]

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
        df = add_moving_averages(df)
        df = add_macd(df)
        df = add_atr(df)
        df = add_pivot_points(df)
        df = add_bearish_candlestick_patterns(df)
        df = add_label_column(
            df,
            annual_expected_return,
            holding_period,
            spread,
            look_ahead_days,
            expected_return_per_trade,
        )
        df = calculate_return_since_last_buy(df)
        df = calculate_days_since_last_buy(df)
        df = df.drop(columns=["1. open", "2. high", "3. low", "label"])

        # Printe the head and tail of the DataFrame
        print(df.head())
        print(df.tail())

        # Load the latest trained model from the live_models folder
        gbm = lgb.Booster(model_file=f"live_models/lightgbm_model_20250209_131215.txt")
        # Load the trained model from the live_models folder depending on the currency pair
        # gbm = lgb.Booster(model_file=f"live_models/lightgbm_model_{from_symbol}.txt")

        # Predict the label for the latest date
        X_latest = df.iloc[-1:]  # Get the latest row
        y_pred_latest = gbm.predict(X_latest, num_iteration=gbm.best_iteration)
        # Convert probabilities to class labels
        y_pred_latest = list(y_pred_latest[0]).index(max(y_pred_latest[0]))

        # Coverting the label to a string
        predicted_label = ""
        if y_pred_latest == 0:
            predicted_label = "buy"
        elif y_pred_latest == 1:
            predicted_label = "hold"
        else:
            predicted_label = "sell"

        # Check the indicator
        # indicator = check_indicator(df)
        # print(f"Indicator for {from_symbol}/{to_symbol}: ", indicator)

        # Append message for the current pair
        full_message += (
            # f"The current RSI indicator for {from_symbol}/{to_symbol} is: {indicator}\n"
            f"The predicted label for {from_symbol}/{to_symbol} is: {predicted_label}\n\n"
        )

    # Send the full notification
    send_notification(full_message)


if __name__ == "__main__":
    main()
