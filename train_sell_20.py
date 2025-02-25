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
    "EUR_TWD",
    "GBP_TWD",
    "AUD_TWD",
    "CHF_TWD",
    "NZD_TWD",
    "JPY_TWD",
]
TRAINING_DATA_YEARS = 10
LOOP_COUNT = 20

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
def find_gross_expected_loss(
    df: pd.DataFrame,
    target_percentage: float = 25.0,
    lookahead: int = 21,
    tolerance: float = 0.5,
    max_iterations: int = 100,
) -> float:
    """
    Binary search for optimal loss threshold that produces target sell percentage.
    """
    lower, upper = 0.0, 0.05  # Search range for loss threshold

    def calculate_sell_percentage(threshold: float) -> float:
        """Helper to calculate sell percentage for given threshold"""
        df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
        sell_pct = (df["future_return"] < -threshold).mean() * 100
        return sell_pct

    for iteration in range(max_iterations):
        current_threshold = (lower + upper) / 2
        sell_percentage = calculate_sell_percentage(current_threshold)

        if abs(sell_percentage - target_percentage) <= tolerance:
            break

        # Adjust search boundaries
        if sell_percentage < target_percentage:
            upper = current_threshold  # Need more sells, make threshold easier
        else:
            lower = current_threshold  # Need fewer sells, make threshold harder

    print(
        f"Optimal loss threshold: {current_threshold:.5f} (Sell%: {sell_percentage:.2f})"
    )
    return current_threshold


def generate_sell_labels(
    df: pd.DataFrame,
    lookahead: int = 20,
    loss_threshold: float = 0.0025,
) -> pd.DataFrame:
    """Create binary sell signals based on future returns"""
    # Calculate future returns
    df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
    df["sell_signal"] = (df["future_return"] < -loss_threshold).astype(int)
    df.drop(columns=["future_return"], inplace=True)
    return df


# ----------------------------
# 🎯 Train LightGBM Model with Hyperparameter Optimization
# ----------------------------
def objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna optimization objective function (updated to match buy version)"""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
        )

        y_pred = (model.predict(X_val) > 0.5).astype(int)
        scores.append(f1_score(y_val, y_pred))  # Focus on F1 score for sell signals

    return np.mean(scores)


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

    # # Plot confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.title("Confusion Matrix")
    # plt.show()

    # # Modified price plot with both signals
    # sell_trades = df[y_pred == 1]
    # real_sell_trades = df[y_true == 1]  # Get actual sell signals

    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df["close"], label="Close Price", alpha=0.7)

    # # Plot predicted signals
    # plt.scatter(
    #     sell_trades.index,
    #     sell_trades["close"],
    #     marker="v",
    #     color="red",
    #     label="Predicted Sell",
    #     alpha=0.8,
    # )

    # # Plot actual signals
    # plt.scatter(
    #     real_sell_trades.index,
    #     real_sell_trades["close"],
    #     marker="^",
    #     color="green",
    #     label="Actual Sell",
    #     alpha=0.8,
    # )

    # plt.legend()
    # plt.title("Predicted vs Actual Sell Signals")
    # plt.show()

    # Return metrics for saving
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def save_artifacts(model, metrics, df, pair, gross_er: float) -> None:
    """Save model and training logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    Path("model_logs").mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = f"saved_models/sell_model_20_{pair}_{timestamp}.pkl"
    joblib.dump(model, model_path)

    # Save logs
    log_entry = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "currency_pair": pair,
                "model_path": model_path,
                "gross_expected_return": gross_er,
                **metrics,
                "dataset_start": df.index.min().strftime("%Y-%m-%d"),
                "dataset_end": df.index.max().strftime("%Y-%m-%d"),
            }
        ]
    )

    log_path = "model_logs/train_sell_20_history.csv"
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
            # Fix timestamp parsing format
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

            # Dynamic threshold finding
            loss_threshold = find_gross_expected_loss(df)
            df = generate_sell_labels(df, loss_threshold=loss_threshold)

            X, y = df[FEATURES], df["sell_signal"]

            # Hyperparameter optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

            # Final model training
            model = lgb.LGBMClassifier(**study.best_params)
            model.fit(X, y)

            metrics = backtest_model(model, df)
            save_artifacts(model, metrics, df, pair, loss_threshold)
