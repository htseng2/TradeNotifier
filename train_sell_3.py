import numpy as np
import pandas as pd
import lightgbm as lgb
import talib
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from datetime import datetime


CURRENCY_PAIRS = [
    "USD_TWD",
    # "EUR_TWD",
    # "GBP_TWD",
    # "AUD_TWD",
    # "CHF_TWD",
    # "NZD_TWD",
    # "JPY_TWD",
]
TRAINING_DATA_YEARS = 10
LOOP_COUNT = 1

FEATURES = [
    "RSI",
    "MACD",
    "MACD_signal",
    "ATR",
    "STOCH_K",
    "STOCH_D",
    "BB_upper",
    "BB_middle",
    "BB_lower",
    "momentum",
    "volatility",
    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_10",
]


# ----------------------------
# 🚀 Load & Preprocess Data
# ----------------------------
def load_data(pair: str) -> pd.DataFrame:
    """Load and preprocess forex data for a currency pair"""
    df = pd.read_csv(
        f"Alpha_Vantage_Data/{pair}.csv",
        skiprows=1,
        header=None,
        names=["timestamp", "open", "high", "low", "close"],
        parse_dates=["timestamp"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df.loc[
        df.index >= pd.Timestamp.now() - pd.DateOffset(years=TRAINING_DATA_YEARS)
    ]


# ----------------------------
# 📈 Feature Engineering
# ----------------------------
def add_technical_indicators(df):
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    df["MACD"], df["MACD_signal"], _ = talib.MACD(
        df["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["STOCH_K"], df["STOCH_D"] = talib.STOCH(
        df["high"],
        df["low"],
        df["close"],
        fastk_period=14,
        slowk_period=3,
        slowd_period=3,
    )
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
        df["close"], timeperiod=20
    )

    # Trend Features
    df["momentum"] = df["close"] - df["close"].shift(5)
    df["volatility"] = df["ATR"] / df["close"]

    # Lag Features
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_{lag}"] = df["close"].pct_change(lag)

    # Drop NaN values
    df.dropna(inplace=True)
    return df


# ----------------------------
# 📊 Define Target (Sell Signal)
# ----------------------------
def define_sell_signal(df):
    df["future_return"] = (
        df["close"].shift(-3) / df["close"] - 1
    )  # Future return in 3 bars
    df["sell_signal"] = (df["future_return"] < -0.0025).astype(
        int
    )  # Threshold for a sell signal
    df.drop(columns=["future_return"], inplace=True)
    return df


# ----------------------------
# 🎯 Train LightGBM Model with Hyperparameter Optimization
# ----------------------------
def train_lightgbm(df):
    features = FEATURES
    target = "sell_signal"

    X, y = df[features], df[target]

    tscv = TimeSeriesSplit(n_splits=5)

    # Define Objective Function for Optuna
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 0.9),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-5, 10),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-5, 10),
        }

        scores = []
        for train_idx, valid_idx in tscv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0),
                ],
            )

            y_pred = (model.predict(X_valid) > 0.5).astype(int)
            scores.append(
                precision_score(y_valid, y_pred)
            )  # Precision for Sell Signals

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params

    print("Best Parameters:", best_params)

    # Final Model Training
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(best_params, train_data, num_boost_round=100)

    # Print features used in training
    print("\n📊 Features used in training:")
    for feature in features:
        print(f" - {feature}")

    return final_model


# ----------------------------
# 📈 Backtest & Evaluate Model
# ----------------------------
def backtest_model(model, df):
    features = FEATURES
    df["sell_prediction"] = model.predict(df[features])

    # Calculate evaluation metrics
    y_true = df["sell_signal"]
    y_pred = (df["sell_prediction"] > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(
        y_true, df["sell_prediction"]
    )  # Use probabilities for ROC AUC

    print(f"\n🔍 Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Modified price plot with both signals
    sell_trades = df[y_pred == 1]
    real_sell_trades = df[y_true == 1]  # Get actual sell signals

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["close"], label="Close Price", alpha=0.7)

    # Plot predicted signals
    plt.scatter(
        sell_trades.index,
        sell_trades["close"],
        marker="v",
        color="red",
        label="Predicted Sell",
        alpha=0.8,
    )

    # Plot actual signals
    plt.scatter(
        real_sell_trades.index,
        real_sell_trades["close"],
        marker="^",
        color="green",
        label="Actual Sell",
        alpha=0.8,
    )

    plt.legend()
    plt.title("Predicted vs Actual Sell Signals")
    plt.show()

    # Return metrics for saving
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def save_artifacts(model, metrics, df, pair):
    """Save model and training logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    Path("model_logs").mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = f"saved_models/sell_model_3_{pair}_{timestamp}.pkl"
    joblib.dump(model, model_path)

    # Save logs
    log_entry = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "currency_pair": pair,
                "model_path": model_path,
                **metrics,
                "dataset_start": df.index.min().strftime("%Y-%m-%d"),
                "dataset_end": df.index.max().strftime("%Y-%m-%d"),
            }
        ]
    )

    log_path = "model_logs/train_sell_3_history.csv"
    log_entry.to_csv(
        log_path, mode="a", header=not Path(log_path).exists(), index=False
    )

    # Clean up old models
    if Path(log_path).exists():
        full_log = pd.read_csv(log_path)
        pair_log = full_log[full_log["currency_pair"] == pair]  # Include current log

        models_to_delete = []
        indices_to_drop = []

        current_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        cutoff_time = current_time - pd.Timedelta(days=1)

        # Fix timestamp comparison in recent entries filter
        recent_entries = pair_log[
            pair_log["timestamp"].apply(lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S"))
            >= cutoff_time
        ]
        # Exclude models with perfect precision from max_f1 calculation
        valid_recent = recent_entries[recent_entries["precision"] < 1.0]
        max_f1 = valid_recent["f1"].max() if not valid_recent.empty else 0

        for index, row in pair_log.iterrows():
            model_time = datetime.strptime(row["timestamp"], "%Y%m%d_%H%M%S")
            age_difference = current_time - model_time

            # Delete if: older than 1 day OR has lower F1 than max in window OR has perfect precision
            if (
                age_difference.days >= 1
                or row["f1"] < max_f1
                or row["precision"] >= 1.0
            ):
                models_to_delete.append(row["model_path"])
                indices_to_drop.append(index)

        # Delete model files
        for model_path in models_to_delete:
            if Path(model_path).exists():
                Path(model_path).unlink()
                print(f"Deleted old model: {model_path}")

        # Update log file if entries were removed
        if indices_to_drop:
            updated_log = full_log.drop(indices_to_drop)
            updated_log.to_csv(log_path, index=False)
            print(f"Removed {len(indices_to_drop)} old entries from log")


# ----------------------------
# 🔥 Main Execution
# ----------------------------
if __name__ == "__main__":
    for _ in range(LOOP_COUNT):
        for pair in CURRENCY_PAIRS:
            df = load_data(pair)
            df = add_technical_indicators(df)
            df = define_sell_signal(df)

            model = train_lightgbm(df)
            metrics = backtest_model(model, df)  # Capture returned metrics
            save_artifacts(model, metrics, df, pair)  # New save call
