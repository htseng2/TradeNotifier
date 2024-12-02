import requests
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def fetch_forex_data(from_symbol, to_symbol):
    import requests

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")  # Replace with your actual API key
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": api_key,
        "outputsize": "full",  # Use "compact" for the last 100 data points
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "Error Message" in data:
        raise ValueError(
            "Error fetching data from Alpha Vantage: " + data["Error Message"]
        )

    return data


def prepare_data_table(data):
    """Convert data into a DataFrame and fill missing values."""
    daily_data = data.get("Time Series FX (Daily)", {})
    df = pd.DataFrame.from_dict(daily_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert closing prices to float
    df["4. close"] = df["4. close"].astype(float)

    # Fill missing data with the previous data
    df = df.fillna(method="ffill")

    return df


def check_indicator(df):
    # Extract the closing prices from the DataFrame
    closing_prices = df["4. close"]

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


def send_email_notification(subject, message, sender_email, receiver_email):
    email_password = os.getenv("EMAIL_PASSWORD")

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
            server.login(sender_email, email_password)
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
    # sender_email = "your_email@gmail.com"
    # receiver_email = "receiver_email@example.com"
    # email_subject = "Trade Notifier Alert"

    # Send notifications
    # send_email_notification(email_subject, message, sender_email, receiver_email)
    send_telegram_notification(message)


def main():
    # Swap the currency pairs to reflect the correct perspective
    currency_pairs = [("JPY", "TWD"), ("USD", "TWD"), ("EUR", "TWD")]

    for from_symbol, to_symbol in currency_pairs:
        data = fetch_forex_data(from_symbol, to_symbol)

        # Prepare the data table
        df = prepare_data_table(data)

        indicator = check_indicator(df)
        print(f"Indicator for {from_symbol}/{to_symbol}: ", indicator)

        # Craft message
        message = f"The current indicator for {from_symbol}/{to_symbol} is: {indicator}"

        # Send notification
        send_notification(message)


if __name__ == "__main__":
    main()
