import requests
import boto3
import pandas as pd
import matplotlib.pyplot as plt


def fetch_forex_data():
    import requests

    api_key = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your actual API key
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": "TWD",
        "to_symbol": "JPY",
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


def send_notification(message):
    # Implement notification logic using AWS SNS or another service
    pass


def main():
    data = fetch_forex_data()
    # Debug by graphing the data
    import pandas as pd
    import matplotlib.pyplot as plt

    # Prepare the data table
    df = prepare_data_table(data)

    # Plot the data
    plt.plot(df.index, df["4. close"])
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title("Forex Data: TWD/JPY")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    indicator = check_indicator(df)
    print(""indicator)


if __name__ == "__main__":
    main()
