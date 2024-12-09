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


def plot_classification(df):
    """Plot the closing prices with classified buy and sell signals."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["4. close"], label="close", color="gray", alpha=0.5)

    # Plot buy signals (label == 0) in green
    buy_signals = df[df["label"] == 0]
    plt.scatter(buy_signals.index, buy_signals["4. close"], color="green", label="Buy")

    # Plot sell signals (label == 2) in red
    sell_signals = df[df["label"] == 2]
    plt.scatter(sell_signals.index, sell_signals["4. close"], color="red", label="Sell")

    plt.legend()
    plt.title("Forex Closing Prices with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


def add_label_column(df, annual_expected_return, holding_period, spread):
    """Add a label column to the DataFrame and prefill with 1."""
    df["label"] = 1

    # Calculate the future return and update the label
    for index in range(len(df) - holding_period[1]):
        current_price = df["4. close"].iloc[index]

        # Check each future price within the holding period
        for future_index in range(index + holding_period[0], index + holding_period[1]):
            days_ahead = future_index - index
            # Calculate the expected return for the specific number of days, including the spread
            expected_return = 1 + (annual_expected_return / 365) * days_ahead + spread

            if df["4. close"].iloc[future_index] > current_price * expected_return:
                df.at[df.index[index], "label"] = 0
                df.at[df.index[future_index], "label"] = 2
                break  # Exit the loop early if a buy signal is found

    return df


def main():
    # Swap the currency pairs to reflect the correct perspective
    # currency_pairs = [("JPY", "TWD"), ("USD", "TWD"), ("EUR", "TWD")]
    currency_pairs = [("JPY", "TWD")]
    annual_expected_return = 0.05
    spread = 0.01  # Spread is transaction cost usually from 0.005 to 0.03
    holding_period = (14, 90)

    for from_symbol, to_symbol in currency_pairs:
        data = fetch_forex_data(from_symbol, to_symbol)

        # Prepare the data table
        df = prepare_data_table(data)

        # Add a column for label either "buy(0)" or "hold(1) or "sell(2)"
        df = add_label_column(df, annual_expected_return, holding_period, spread)

        # Plot the data with classification
        plot_classification(df)

        # Check the indicator
        indicator = check_indicator(df)
        print(f"Indicator for {from_symbol}/{to_symbol}: ", indicator)

        # Craft message
        message = f"The current indicator for {from_symbol}/{to_symbol} is: {indicator}"

        # Send notification
        send_notification(message)


if __name__ == "__main__":
    main()
