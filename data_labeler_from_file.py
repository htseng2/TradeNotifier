from numpy import save
import pandas as pd
import matplotlib.pyplot as plt

# Constants
spread_table = {
    ("USD", "TWD"): 0.003,
    ("CNY", "TWD"): 0.003,
    ("EUR", "TWD"): 0.012,
    ("NZD", "TWD"): 0.012,
    ("SGD", "TWD"): 0.013,
    ("GBP", "TWD"): 0.013,
    ("AUD", "TWD"): 0.014,
    ("CHF", "TWD"): 0.015,
    ("HKD", "TWD"): 0.017,
    ("CAD", "TWD"): 0.017,
    ("DKK", "TWD"): 0.018,
    ("JPY", "TWD"): 0.02,
    # Not considering the following pairs for now
    ("SEK", "TWD"): 0.03,
    ("THB", "TWD"): 0.032,
    ("ZAR", "TWD"): 0.055,
    ("TRY", "TWD"): 0.48,
}


def fetch_forex_data_from_file(file_path):
    return pd.read_csv(file_path)


def prepare_data_table(df):
    # Convert 'Date' column to datetime format
    df["Date"] = pd.to_datetime(pd.to_datetime(df["Date"]).dt.date)

    # Set 'Date' column as the index
    df.set_index("Date", inplace=True)
    df = df.sort_index()

    # Convert closing prices to float
    df["Close"] = df["Close"].astype(float)

    # Remove weekends
    df = df[df.index.dayofweek < 5]  # 0=Monday, 4=Friday

    return df


def add_moving_averages(df):
    """Add moving averages to the DataFrame."""
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    return df


def add_macd(df):
    """Add MACD to the DataFrame."""
    df["MACD"] = (
        df["Close"].ewm(span=12, adjust=False).mean()
        - df["Close"].ewm(span=26, adjust=False).mean()
    )
    return df


def add_rsi(df, window=14):
    """Add RSI (Relative Strength Index) to the DataFrame."""
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_stoch(df):
    """Add stochastic oscillator to the DataFrame."""
    df["Stoch"] = (df["Close"] - df["Min_14"]) / (df["Max_14"] - df["Min_14"])
    return df


def add_bbands(df):
    """Add Bollinger Bands to the DataFrame."""
    df["BBands"] = df["Close"].rolling(window=20).mean()
    return df


def add_atr(df):
    """Add Average True Range to the DataFrame."""
    df["ATR"] = df["Close"].rolling(window=14).mean()
    return df


def add_daily_return(df):
    """Add daily return to the DataFrame."""
    df["Daily_Return"] = df["Close"].pct_change()
    return df


def add_weekly_return(df):
    """Add weekly return to the DataFrame."""
    df["Weekly_Return"] = df["Close"].pct_change(periods=5)
    return df


def add_max_min(df):
    """Add maximum and minimum values over 14, 50, and 90 days to the DataFrame."""
    df["Max_14"] = df["Close"].rolling(window=14).max()
    df["Min_14"] = df["Close"].rolling(window=14).min()
    df["Max_50"] = df["Close"].rolling(window=50).max()
    df["Min_50"] = df["Close"].rolling(window=50).min()
    df["Max_90"] = df["Close"].rolling(window=90).max()
    df["Min_90"] = df["Close"].rolling(window=90).min()
    return df


def add_label_column(df, annual_expected_return, holding_period, spread):
    """Add a label column to the DataFrame and prefill with 1."""
    df["label"] = 1

    # Calculate the future return and update the label
    for index in range(len(df) - holding_period[1]):
        current_price = df["Close"].iloc[index]

        LOOK_AHEAD_DAYS = 21
        EXPECTED_RETURN_PER_TRADE = 0.005

        # Check each future price within the holding period
        for future_index in range(index, index + LOOK_AHEAD_DAYS):
            # Calculate the threshold (expected return) for the specific number of days, including the spread
            expected_return = 1 + EXPECTED_RETURN_PER_TRADE + spread
            threshold = current_price * expected_return
            if df["Close"].iloc[future_index] > threshold:
                df.at[df.index[index], "label"] = 0
                # df.at[df.index[future_index], "label"] = 2
                break  # Exit the loop early if a buy signal is found

            # If all the future prices (now + LOOK_AHEAD_DAYS) are below the current price, set the label to 2
            is_all_below_threshold = all(
                df["Close"].iloc[future_index] <= current_price
                for future_index in range(index, index + LOOK_AHEAD_DAYS)
            )
            if is_all_below_threshold:
                df.at[df.index[index], "label"] = 2
                break  # Exit the loop early if a sell signal is found

    return df


def plot_classification(df):
    """Plot the closing prices with classified buy and sell signals."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label="close", color="gray", alpha=0.5)

    # Plot buy signals (label == 0) in green
    buy_signals = df[df["label"] == 0]
    plt.scatter(buy_signals.index, buy_signals["Close"], color="green", label="Buy")

    # Plot sell signals (label == 2) in red
    sell_signals = df[df["label"] == 2]
    plt.scatter(sell_signals.index, sell_signals["Close"], color="red", label="Sell")

    # Plot moving averages
    # plt.plot(df.index, df["MA_14"], label="MA 14", color="orange")
    # plt.plot(df.index, df["MA_50"], label="MA 50", color="blue")
    # plt.plot(df.index, df["MA_90"], label="MA 90", color="purple")

    plt.legend()
    plt.title("Forex Closing Prices with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


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
    # currency_pairs = [("JPY", "TWD")]
    annual_expected_return = 0.20
    spread = 0.02  # Spread is transaction cost usually from 0.005 to 0.03
    holding_period = (14, 90)

    for from_symbol, to_symbol in currency_pairs:
        file_path = f"Forex/Forex - {from_symbol}.csv"
        df = fetch_forex_data_from_file(file_path)

        # # Prepare the data table
        df = prepare_data_table(df)

        # # Add maximum and minimum values
        df = add_max_min(df)

        # # Add moving averages (Total Indicators: 2)
        df = add_moving_averages(df)

        # # Add MACD (Total Indicators: 3)
        df = add_macd(df)

        # # Add RSI (Total Indicators: 4)
        df = add_rsi(df)

        # # Add stochastic oscillator (Total Indicators: 5)
        df = add_stoch(df)

        # # Add Bollinger Bands (Total Indicators: 6)
        df = add_bbands(df)

        # # Add Average True Range (Total Indicators: 7)
        df = add_atr(df)

        # # Add daily return (Total Indicators: 8)
        df = add_daily_return(df)

        # # Add weekly return (Total Indicators: 9)
        df = add_weekly_return(df)

        # # Add a column for label either "buy(0)" or "hold(1) or "sell(2)"
        df = add_label_column(df, annual_expected_return, holding_period, spread)

        # # Chop off the first and last days based on the longest holding period
        longest_holding_period = holding_period[-1]
        df = df.iloc[longest_holding_period:-longest_holding_period]

        # Save the DataFrame to a CSV file
        save_path = f"labeled_data/labeled_data_{from_symbol}.csv"
        df.to_csv(save_path, index=False)

        # # Print head and tail of the DataFrame
        print(df.head())
        print(df.tail())

        # Plot the data with classification
        # plot_classification(df)

    # Combine all the labeled data and output to a single CSV file
    combined_df = pd.concat(
        [
            pd.read_csv(f"labeled_data/labeled_data_{from_symbol}.csv")
            for from_symbol, _ in currency_pairs
        ]
    )
    combined_df.to_csv("labeled_data/labeled_data.csv", index=False)


if __name__ == "__main__":
    main()
