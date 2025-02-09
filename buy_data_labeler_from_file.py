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


def add_bearish_candlestick_patterns(df):
    """Add bearish candlestick patterns to the DataFrame."""
    df["Bearish_Candlestick_Patterns"] = df["Close"].rolling(window=3).min()
    return df


def calculate_return_since_last_buy(df):
    """Calculate the return since the last buy signal (label == 0) and add it to the DataFrame."""
    last_buy_index = None
    return_since_buy = []

    for index, row in df.iterrows():
        if row["buy"] == 1:
            if last_buy_index is not None:
                buy_price = df.at[last_buy_index, "Close"]
                current_price = row["Close"]
                return_since_buy.append((current_price - buy_price) / buy_price)
            else:
                return_since_buy.append(0)
            last_buy_index = index
        elif last_buy_index is not None:
            buy_price = df.at[last_buy_index, "Close"]
            current_price = row["Close"]
            return_since_buy.append((current_price - buy_price) / buy_price)
        else:
            return_since_buy.append(0)  # No return if no buy signal has occurred yet

    df["Return_Since_Last_Buy"] = return_since_buy
    return df


def calculate_days_since_last_buy(df):
    """Calculate the number of days since the last buy signal (label == 0) and add it to the DataFrame."""
    last_buy_index = None
    days_since_buy = []

    for index, row in df.iterrows():
        if row["buy"] == 1:
            if last_buy_index is not None:
                days_since_buy.append((index - last_buy_index).days)
            else:
                days_since_buy.append(0)
            last_buy_index = index
        elif last_buy_index is not None:
            days_since_buy.append((index - last_buy_index).days)
        else:
            days_since_buy.append(0)

    df["Days_Since_Last_Buy"] = days_since_buy
    return df


def add_pivot_points(df):
    """Add Pivot Points / Swing High-Low Features to the DataFrame."""
    df["Pivot_Points"] = df["Close"].rolling(window=3).max()
    return df


def add_buy_sell_column(
    df,
    annual_expected_return,
    holding_period,
    spread,
    look_ahead_days,
    expected_return_per_trade,
):
    """Add a label column to the DataFrame and prefill with 1."""
    df["buy"] = 0
    df["sell"] = 0
    # Calculate the future return and update the label
    for index in range(len(df) - holding_period[1]):
        current_price = df["Close"].iloc[index]

        # Check each future price within the holding period
        for future_index in range(index, index + look_ahead_days):
            # Calculate the threshold (expected return) for the specific number of days, including the spread
            expected_return = 1 + expected_return_per_trade + spread
            threshold = current_price * expected_return
            if df["Close"].iloc[future_index] > threshold:
                df.at[df.index[index], "buy"] = 1

            # If all the future prices (now + LOOK_AHEAD_DAYS) are below the current price, set the label to 1
            is_all_below_threshold = all(
                df["Close"].iloc[future_index] <= current_price
                for future_index in range(index, index + look_ahead_days)
            )
            if is_all_below_threshold:
                df.at[df.index[index], "sell"] = 1

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
    holding_period = (1, 60)
    look_ahead_days = 21
    expected_return_per_trade = 0.005

    for from_symbol, to_symbol in currency_pairs:
        file_path = f"Forex/Forex - {from_symbol}.csv"
        df = fetch_forex_data_from_file(file_path)

        # Prepare the data table
        df = prepare_data_table(df)

        # Add maximum and minimum values ✅
        df = add_max_min(df)

        # Add moving averages (Total Indicators: 2) ✅
        df = add_moving_averages(df)

        # Add MACD (Total Indicators: 3) ✅
        df = add_macd(df)

        # Add Average True Range (Total Indicators: 7) ✅
        df = add_atr(df)

        # Add Pivot Points / Swing High-Low Features (Total Indicators: 8) ✅
        df = add_pivot_points(df)

        # Add bearish candlestick patterns (Total Indicators: 9) ✅
        df = add_bearish_candlestick_patterns(df)

        # Add a column for label either "buy(0)" or "hold(1) or "sell(2)"
        df = add_buy_sell_column(
            df,
            annual_expected_return,
            holding_period,
            spread,
            look_ahead_days,
            expected_return_per_trade,
        )

        # Add the return since the last buy signal (Total Indicators: 10) ✅
        df = calculate_return_since_last_buy(df)

        # Add the days since the last buy signal (Total Indicators: 11) ✅
        df = calculate_days_since_last_buy(df)

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
