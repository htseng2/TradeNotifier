import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # Changed from hardcoded "demo"
SYMBOLS = [
    # "AAPL",
    # "MSFT",
    "MRK",
    "GOOG",
    "WMT",
    "COST",
    "XOM",
    "PG",
    "JNJ",
    "CAT",
]

# Loop through each symbol
for SYMBOL in SYMBOLS:
    # Construct the API URL for daily adjusted data
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={API_KEY}&outputsize=full&datatype=json"

    # Make the API request
    response = requests.get(url)
    print(f"Downloading data for {SYMBOL}...")
    data = response.json()

    # Extract the time series data
    time_series = data.get("Time Series (Daily)", {})

    # Convert to pandas DataFrame and clean up
    df = pd.DataFrame.from_dict(time_series, orient="index")
    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        }
    )
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().astype(float)

    # Save to CSV
    df.to_csv(f"Alpha_Vantage_Stock/{SYMBOL}_daily_stock_data.csv")
    print(f"Downloaded {len(df)} days of data for {SYMBOL}\n")
