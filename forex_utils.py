import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def fetch_forex_data(from_symbol, to_symbol):
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
    """Convert data into a DataFrame and remove weekends."""
    daily_data = data.get("Time Series FX (Daily)", {})
    df = pd.DataFrame.from_dict(daily_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert closing prices to float
    df["4. close"] = df["4. close"].astype(float)
    df.rename(columns={"4. close": "Close"}, inplace=True)

    # Remove weekends
    df = df[df.index.dayofweek < 5]  # 0=Monday, 4=Friday

    # Preview head and tail
    print(df.head())
    print(df.tail())

    return df
