from forex_utils import fetch_forex_data, prepare_data_table
import matplotlib.pyplot as plt
import pandas as pd  # Add pandas import if not already present


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


def add_moving_averages(df):
    """Add moving averages to the DataFrame."""
    df["MA_14"] = df["4. close"].rolling(window=14).mean()
    df["MA_50"] = df["4. close"].rolling(window=50).mean()
    df["MA_90"] = df["4. close"].rolling(window=90).mean()
    return df


def add_max_min(df):
    """Add maximum and minimum values over 14, 50, and 90 days to the DataFrame."""
    df["Max_14"] = df["4. close"].rolling(window=14).max()
    df["Min_14"] = df["4. close"].rolling(window=14).min()
    df["Max_50"] = df["4. close"].rolling(window=50).max()
    df["Min_50"] = df["4. close"].rolling(window=50).min()
    df["Max_90"] = df["4. close"].rolling(window=90).max()
    df["Min_90"] = df["4. close"].rolling(window=90).min()
    return df


def add_rsi(df, window=14):
    """Add RSI (Relative Strength Index) to the DataFrame."""
    delta = df["4. close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


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
    # currency_pairs = [("JPY", "TWD"), ("USD", "TWD"), ("EUR", "TWD")]
    currency_pairs = [("JPY", "TWD")]
    annual_expected_return = 0.05
    spread = 0.01  # Spread is transaction cost usually from 0.005 to 0.03
    holding_period = (14, 90)

    for from_symbol, to_symbol in currency_pairs:
        data = fetch_forex_data(from_symbol, to_symbol)

        # Prepare the data table
        df = prepare_data_table(data)

        # Add moving averages
        df = add_moving_averages(df)

        # Add maximum and minimum values
        df = add_max_min(df)

        # Add RSI
        df = add_rsi(df)

        # Add a column for label either "buy(0)" or "hold(1) or "sell(2)"
        df = add_label_column(df, annual_expected_return, holding_period, spread)

        # Chop off the first and last days based on the longest holding period
        longest_holding_period = holding_period[-1]
        df = df.iloc[longest_holding_period:-longest_holding_period]

        # Drop columns "1. open", "2. high", "3. low"
        df = df.drop(columns=["1. open", "2. high", "3. low"])

        # Save the DataFrame to a CSV file
        df.to_csv("labeled_data.csv", index=False)

        # Preview the data table head and tail
        print(df.head())
        print(df.tail())

        # Plot the data with classification
        plot_classification(df)

        # You can add more processing or save the labeled data here


if __name__ == "__main__":
    main()
