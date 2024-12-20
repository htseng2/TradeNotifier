from cProfile import label
from data_labeler import add_max_min, add_moving_averages, add_rsi
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
    # Swap the currency pairs to reflect the correct perspective
    currency_pairs = [
        ("USD", "TWD"),
        ("CNY", "TWD"),
        ("EUR", "TWD"),
        ("NZD", "TWD"),
        ("SGD", "TWD"),
        ("GBP", "TWD"),
        ("AUD", "TWD"),
        ("CHF", "TWD"),
        ("HKD", "TWD"),
        ("CAD", "TWD"),
        ("DKK", "TWD"),
        ("JPY", "TWD"),
    ]

    # Initialize an empty message
    full_message = ""

    for from_symbol, to_symbol in currency_pairs:
        data = fetch_forex_data(from_symbol, to_symbol)

        # Prepare the data table
        df = prepare_data_table(data)

        # Enhance the DataFrame with additional features
        df = add_moving_averages(df)
        df = add_max_min(df)
        df = add_rsi(df)
        df = df.drop(columns=["1. open", "2. high", "3. low"])

        # Printe the head and tail of the DataFrame
        print(df.head())
        print(df.tail())

        # Load the trained model
        gbm = lgb.Booster(model_file="lightgbm_model.txt")

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
        indicator = check_indicator(df)
        print(f"Indicator for {from_symbol}/{to_symbol}: ", indicator)

        # Append message for the current pair
        full_message += (
            f"The current RSI indicator for {from_symbol}/{to_symbol} is: {indicator}\n"
            f"The predicted label for the latest date is: {predicted_label}\n\n"
        )

    # Send the full notification
    send_notification(full_message)


if __name__ == "__main__":
    main()
