import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def fetch_forex_data(from_symbol, to_symbol):
    """Fetch FX daily data from Alpha Vantage API"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": api_key,
        "outputsize": "full",
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"API Error: {data['Error Message']}")
    return data


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
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
        }
    )[["Open", "High", "Low", "Close"]]

    # Filter out weekends
    return df[df.index.dayofweek < 5]


def main():
    currency_pairs = [
        # ("USD", "TWD"),
        # ("EUR", "TWD"),
        # ("SGD", "TWD"),
        # ("GBP", "TWD"),
        # ("AUD", "TWD"),
        # ("CHF", "TWD"),
        # ("CAD", "TWD"),
        # ("JPY", "TWD"),
        # ("HKD", "TWD"),
        ("NZD", "TWD"),
        # Not considering the following pairs for now
        # ("CNY", "TWD"),
        # ("THB", "TWD"),
        # ("ZAR", "TWD"),
    ]

    # Create directory if it doesn't exist
    os.makedirs("Alpha_Vantage_Data", exist_ok=True)

    for from_currency, to_currency in currency_pairs:
        try:
            # Fetch data using existing utility function
            raw_data = fetch_forex_data(from_currency, to_currency)

            # Prepare data using existing utility function
            df = prepare_data_table(raw_data)

            # Rename columns to match required format (preserved as requested)
            df = df.rename(
                columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "Close": "Close",
                }
            )[["Open", "High", "Low", "Close"]]

            # Save to CSV (with append functionality)
            filename = f"Alpha_Vantage_Data/{from_currency}_{to_currency}.csv"

            if os.path.exists(filename):
                # Read existing data and filter new entries
                existing_df = pd.read_csv(filename, index_col=0, parse_dates=True)
                new_data = df[~df.index.isin(existing_df.index)]
                combined_df = pd.concat([existing_df, new_data]).sort_index()
            else:
                combined_df = df

            combined_df.to_csv(filename)
            print(
                f"Updated {filename} with {len(combined_df)} total records ({len(new_data) if os.path.exists(filename) else len(df)} new)"
            )

        except Exception as e:
            print(f"Error processing {from_currency}/{to_currency}: {str(e)}")


if __name__ == "__main__":
    main()
