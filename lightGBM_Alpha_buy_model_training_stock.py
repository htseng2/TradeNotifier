import pandas as pd
import matplotlib.pyplot as plt
import optuna
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
)
import numpy as np
import datetime
import seaborn as sns
import csv
import json
from pathlib import Path


# ----------------------------
# Data Preparation Functions
# ----------------------------
def fetch_forex_data(file_path):
    # Read CSV with date as index column
    return pd.read_csv(file_path, index_col=0)


def fetch_stock_data(file_path):
    """Read stock CSV with correct column names"""
    df = pd.read_csv(
        file_path,
        header=0,  # Use the first row as header
        names=["date", "Open", "High", "Low", "Close", "Volume"],  # Map columns
        parse_dates=["date"],
        index_col="date",
    )
    return df


def prepare_data_table(df):
    # Convert existing index to datetime
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
    df["Close"] = df["Close"].astype(float)
    return df[df.index.dayofweek < 5]


def add_technical_indicators(df):
    # Moving Averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f"MA_{window}"] = df["Close"].rolling(window).mean()

    # Price Extremes
    for window in [5, 10, 20, 50, 100, 200]:
        df[f"Max_{window}"] = df["Close"].rolling(window).max()
        df[f"Min_{window}"] = df["Close"].rolling(window).min()

    # Momentum Indicators
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    # Stochastic Oscillator (STOCH)
    low_9 = df["Low"].rolling(9).min()
    high_9 = df["High"].rolling(9).max()
    df["STOCH_%K"] = 100 * ((df["Close"] - low_9) / (high_9 - low_9))
    df["STOCH_%D"] = df["STOCH_%K"].rolling(6).mean()

    # Stochastic RSI
    rsi = df["RSI"].rolling(14, min_periods=14)
    df["STOCHRSI_%K"] = (df["RSI"] - rsi.min()) / (rsi.max() - rsi.min()) * 100
    df["STOCHRSI_%D"] = df["STOCHRSI_%K"].rolling(3).mean()

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ADX
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14).mean() / tr.ewm(alpha=1 / 14).mean())
    minus_di = 100 * (minus_dm.ewm(alpha=1 / 14).mean() / tr.ewm(alpha=1 / 14).mean())
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df["ADX"] = dx.ewm(alpha=1 / 14).mean()

    # Williams %R
    high_14 = df["High"].rolling(14).max()
    low_14 = df["Low"].rolling(14).min()
    df["Williams_%R"] = (high_14 - df["Close"]) / (high_14 - low_14) * -100

    # CCI
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["CCI"] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std())

    # ATR
    df["ATR"] = tr.rolling(14).mean()

    # Highs/Lows
    df["High_14"] = df["High"].rolling(14).max()
    df["Low_14"] = df["Low"].rolling(14).min()

    # Ultimate Oscillator
    bp = df["Close"] - pd.concat([df["Low"], df["Close"].shift()], axis=1).min(axis=1)
    tr = pd.concat([df["High"], df["Close"].shift()], axis=1).max(axis=1) - pd.concat(
        [df["Low"], df["Close"].shift()], axis=1
    ).min(axis=1)

    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    df["Ultimate_Osc"] = 100 * (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)

    # ROC
    df["ROC"] = df["Close"].pct_change(14) * 100

    # Bull/Bear Power
    ema13 = df["Close"].ewm(span=13, adjust=False).mean()
    df["Bull_Power"] = df["High"] - ema13
    df["Bear_Power"] = df["Low"] - ema13

    # Pivot Points (Daily)
    prev_day = df.shift(1)
    df["Pivot"] = (prev_day["High"] + prev_day["Low"] + prev_day["Close"]) / 3
    df["S1"] = (2 * df["Pivot"]) - prev_day["High"]
    df["R1"] = (2 * df["Pivot"]) - prev_day["Low"]
    df["S2"] = df["Pivot"] - (prev_day["High"] - prev_day["Low"])
    df["R2"] = df["Pivot"] + (prev_day["High"] - prev_day["Low"])
    df["S3"] = prev_day["Low"] - 2 * (prev_day["High"] - df["Pivot"])
    df["R3"] = prev_day["High"] + 2 * (df["Pivot"] - prev_day["Low"])

    # Fibonacci Pivot Points
    pp = (prev_day["High"] + prev_day["Low"] + prev_day["Close"]) / 3
    df["Fib_S1"] = pp - (0.382 * (prev_day["High"] - prev_day["Low"]))
    df["Fib_S2"] = pp - (0.618 * (prev_day["High"] - prev_day["Low"]))
    df["Fib_R1"] = pp + (0.382 * (prev_day["High"] - prev_day["Low"]))
    df["Fib_R2"] = pp + (0.618 * (prev_day["High"] - prev_day["Low"]))

    # Camarilla Pivot Points
    df["Camarilla_R3"] = (
        prev_day["Close"] + (prev_day["High"] - prev_day["Low"]) * 1.1 / 4
    )
    df["Camarilla_R2"] = (
        prev_day["Close"] + (prev_day["High"] - prev_day["Low"]) * 1.1 / 6
    )
    df["Camarilla_R1"] = (
        prev_day["Close"] + (prev_day["High"] - prev_day["Low"]) * 1.1 / 12
    )
    df["Camarilla_S1"] = (
        prev_day["Close"] - (prev_day["High"] - prev_day["Low"]) * 1.1 / 12
    )
    df["Camarilla_S2"] = (
        prev_day["Close"] - (prev_day["High"] - prev_day["Low"]) * 1.1 / 6
    )
    df["Camarilla_S3"] = (
        prev_day["Close"] - (prev_day["High"] - prev_day["Low"]) * 1.1 / 4
    )

    # Added feature
    df["MA_10_50_ratio"] = df["MA_10"] / df["MA_50"]
    df["MA_50_200_ratio"] = df["MA_50"] / df["MA_200"]

    # Derived features like using (Min_200, Max_200), (Min_100, Max_100), (Min_50, Max_50)
    df["Min_200_Max_200_ratio"] = df["Min_200"] / df["Max_200"]
    df["Min_100_Max_100_ratio"] = df["Min_100"] / df["Max_100"]
    df["Min_50_Max_50_ratio"] = df["Min_50"] / df["Max_50"]

    return df


def generate_labels(
    df, lookahead=21, expected_return=0.075, spread=0.02, trim_data=True
):
    """Generate trading labels while handling lookahead bias

    Args:
        trim_data (bool): If True, removes incomplete windows at start/end.
            Set to False to keep all dates (for testing/analysis)
    """
    df["buy"] = 0
    threshold = 1 + expected_return + spread

    for i in range(len(df) - lookahead):
        current_price = df["Close"].iloc[i]
        future_prices = df["Close"].iloc[i + 1 : i + lookahead + 1]

        if any(future_prices > current_price * threshold):
            df.at[df.index[i], "buy"] = 1

    return df.iloc[lookahead:-lookahead] if trim_data else df


# ----------------------------
# Model Training Functions
# ----------------------------
def prepare_features(df):
    # Explicit feature selection (validated against add_technical_indicators)
    selected_features = [
        # Price data
        "Close",
        "Open",
        "High",
        "Low",
        # Volume data
        "Volume",
        # Core technical indicators
        "ATR",
        "ADX",
        "RSI",
        "STOCH_%K",
        "STOCH_%D",
        "STOCHRSI_%K",
        "STOCHRSI_%D",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "Williams_%R",
        "CCI",
        "Ultimate_Osc",
        "ROC",
        # Moving averages
        "MA_5",
        "MA_10",
        "MA_20",
        "MA_50",
        "MA_100",
        "MA_200",
        # Price extremes (all windows)
        "Max_5",
        "Min_5",
        "Max_10",
        "Min_10",
        "Max_20",
        "Min_20",
        "Max_50",
        "Min_50",
        "Max_100",
        "Min_100",
        "Max_200",
        "Min_200",
        # Ratios
        "MA_10_50_ratio",
        "MA_50_200_ratio",
        "Min_50_Max_50_ratio",
        "Min_100_Max_100_ratio",
        "Min_200_Max_200_ratio",
        # Pivot points
        "Pivot",
        "S1",
        "R1",
        "S2",
        "R2",
        "S3",
        "R3",
        # Fibonacci levels
        "Fib_R1",
        "Fib_S1",
        "Fib_R2",
        "Fib_S2",
        # Camarilla levels
        "Camarilla_R1",
        "Camarilla_S1",
        "Camarilla_R2",
        "Camarilla_S2",
        "Camarilla_R3",
        "Camarilla_S3",
        # Power indicators
        "Bull_Power",
        "Bear_Power",
        # Additional features
        "High_14",
        "Low_14",
    ]

    # Filter to only existing columns
    valid_features = [f for f in selected_features if f in df.columns]
    return df[valid_features], df["buy"]


def optimize_hyperparameters(X_train, y_train, X_valid, y_valid):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "min_sum_hessian_in_leaf": trial.suggest_int(
                "min_sum_hessian_in_leaf", 5, 20
            ),
        }

        model = lgb.train(
            params,
            lgb.Dataset(X_train, y_train),
            valid_sets=[lgb.Dataset(X_valid, y_valid)],
            callbacks=[early_stopping(10)],
        )
        return log_loss(y_valid, model.predict(X_valid))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    return study.best_params


def train_model(X_train, y_train, X_valid, y_valid, best_params):
    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopping(10)]
    )
    return model


def plot_learning_curve(
    estimator, title, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5)
):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("ROC AUC Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring="roc_auc"
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")
    plt.show()


def save_results_report(results_path, metrics, features, params):
    """Append model results to a CSV file with structured data"""
    Path(results_path).parent.mkdir(exist_ok=True, parents=True)

    # Prepare data row
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
        "features": "|".join(sorted(features)),
        "num_features": len(features),
        "parameters": json.dumps(params, sort_keys=True),
    }

    # Write/append to CSV
    file_exists = Path(results_path).exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------
# Main Execution Flow
# ----------------------------
def main():
    # Data Preparation
    stock_symbols = [
        "AAPL",
        # "MSFT",
        # "MRK",
        # "GOOG",
        # "WMT",
        # "COST",
        # "XOM",
        # "PG",
        # "JNJ",
        # "CAT",
    ]  # Single symbol instead of currency pairs

    dfs = []
    for symbol in stock_symbols:
        # Update path to match stock data format
        df = fetch_stock_data(f"Alpha_Vantage_Data/{symbol}_daily_stock_data.csv")
        df = prepare_data_table(df)
        df = add_technical_indicators(df)
        df = generate_labels(df)
        dfs.append(df)

    full_df = pd.concat(dfs)

    # Model Training
    X, y = prepare_features(full_df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_params = optimize_hyperparameters(X_train, y_train, X_valid, y_valid)
    model = train_model(X_train, y_train, X_valid, y_valid, best_params)

    # Learning Curve Analysis
    plot_learning_curve(model, "Learning Curve (ROC AUC)", X_train, y_train)

    # Evaluation
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict(X_valid)
    val_proba = model.predict_proba(X_valid)[:, 1]

    print("\n=== Training Set Performance ===")
    print(classification_report(y_train, train_pred))
    print(f"Train ROC AUC: {roc_auc_score(y_train, train_proba):.2%}")

    print("\n=== Validation Set Performance ===")
    print(classification_report(y_valid, val_pred))
    print(f"Validation ROC AUC: {roc_auc_score(y_valid, val_proba):.2%}")

    print("\nConfusion Matrix (Validation):")
    print(confusion_matrix(y_valid, val_pred))

    # Enhanced evaluation metrics
    print("\n=== Bias-Variance Indicators ===")
    # Calculate log loss for both sets
    train_log_loss = log_loss(y_train, train_proba)
    val_log_loss = log_loss(y_valid, val_proba)
    print(
        f"Train Log Loss: {train_log_loss:.4f} | Validation Log Loss: {val_log_loss:.4f}"
    )

    # Calculate ROC AUC gap
    roc_gap = roc_auc_score(y_train, train_proba) - roc_auc_score(y_valid, val_proba)
    print(f"ROC AUC Gap (Train - Val): {roc_gap:.2%}")

    # Add training set confusion matrix
    print("\nConfusion Matrix (Training):")
    print(confusion_matrix(y_train, train_pred))

    # Feature Importance Analysis
    print("\n=== Feature Importance ===")
    ax = lgb.plot_importance(
        model.booster_,
        max_num_features=20,
        importance_type="gain",
        title="Feature Importance (Gain)",
        figsize=(10, 6),
    )
    plt.show()

    # Get and display top features
    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values(by="Importance", ascending=False)

    print("\nTop 20 Features by Gain Importance:")
    print(importance.head(20).to_string(index=False))

    # Save Model first to get filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/buy_model_{timestamp}.txt"
    model.booster_.save_model(model_filename)

    # Save results to history (now includes model filename)
    metrics = {
        "precision_0": float(
            classification_report(y_valid, val_pred, output_dict=True)["0"]["precision"]
        ),
        "recall_0": float(
            classification_report(y_valid, val_pred, output_dict=True)["0"]["recall"]
        ),
        "f1_0": float(
            classification_report(y_valid, val_pred, output_dict=True)["0"]["f1-score"]
        ),
        "precision_1": float(
            classification_report(y_valid, val_pred, output_dict=True)["1"]["precision"]
        ),
        "recall_1": float(
            classification_report(y_valid, val_pred, output_dict=True)["1"]["recall"]
        ),
        "f1_1": float(
            classification_report(y_valid, val_pred, output_dict=True)["1"]["f1-score"]
        ),
        "roc_auc": roc_auc_score(y_valid, val_proba),
        "train_log_loss": train_log_loss,
        "val_log_loss": val_log_loss,
        "roc_auc_gap": roc_gap,
        "model_filename": model_filename,
        "symbol": symbol,
    }

    save_results_report(
        results_path="lightGBM_stock_model_history.csv",
        metrics=metrics,
        features=list(X.columns),
        params=best_params,
    )

    # Test on first currency pair
    first_pair = stock_symbols[0]
    test_df = fetch_stock_data(f"Alpha_Vantage_Data/{first_pair}_daily_stock_data.csv")
    test_df = prepare_data_table(test_df)
    test_df = add_technical_indicators(test_df)
    test_df = generate_labels(test_df, trim_data=False)

    # Prepare features and filter
    X_test, y_test = prepare_features(test_df)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)  # Align columns

    # Predict
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(test_df.index, test_df["Close"], label="Price", alpha=0.5)

    # True buy signals
    true_buys = test_df[test_df["buy"] == 1]
    plt.scatter(
        true_buys.index,
        true_buys["Close"],
        color="green",
        marker="^",
        s=100,
        label="True Buy Signals",
    )

    # Predicted buy signals
    predicted_buys = test_df.loc[X_test.index[test_pred == 1]]
    plt.scatter(
        predicted_buys.index,
        predicted_buys["Close"],
        color="red",
        marker="v",
        s=100,
        label="Predicted Buy Signals",
    )

    plt.title(f"{first_pair} Buy Signal Comparison")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
