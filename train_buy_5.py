import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import talib
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

CURRENCY_PAIRS = [
    "USD_TWD",
    # "EUR_TWD",
    # "GBP_TWD",
    # "AUD_TWD",
    # "CHF_TWD",
    # "NZD_TWD",
    # "JPY_TWD",
]
LOOP_COUNT = 20
TRAINING_DATA_YEARS = 10
FEATURES = ["SMA_50", "SMA_200", "RSI", "MACD", "BB_upper", "BB_lower", "ATR"]


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


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical indicators for trading signals"""
    df["SMA_50"] = talib.SMA(df["close"], timeperiod=50)
    df["SMA_200"] = talib.SMA(df["close"], timeperiod=200)
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)

    macd, signal, _ = talib.MACD(
        df["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd - signal

    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
    df["BB_upper"] = upper
    df["BB_lower"] = lower
    df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

    return df.dropna()


def generate_labels(df: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    """Create binary buy signals based on future returns"""
    df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
    df["buy_signal"] = (df["future_return"] > 0.002).astype(int)
    return df.drop(columns=["future_return"])


def objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna optimization objective function"""
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
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )

        y_pred = (model.predict(X_val) > 0.5).astype(int)
        metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        metrics["precision"].append(precision_score(y_val, y_pred))
        metrics["recall"].append(recall_score(y_val, y_pred))
        metrics["f1"].append(f1_score(y_val, y_pred))
        metrics["roc_auc"].append(roc_auc_score(y_val, y_pred))

    return np.mean(metrics["accuracy"])


def train_final_model(X: pd.DataFrame, y: pd.Series, best_params: dict) -> tuple:
    """Train and evaluate final model with best parameters"""
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)

    predictions = model.predict(X)
    return model, {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1": f1_score(y, predictions),
        "roc_auc": roc_auc_score(y, predictions),
    }


def backtest_model(model, data: pd.DataFrame, pair: str) -> dict:
    """Backtest model and generate performance metrics"""
    features = FEATURES
    data["predicted_buy_signal"] = model.predict(data[features])
    y_true = data["buy_signal"]
    y_pred = data["predicted_buy_signal"]

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }

    print(f"\n🔍 Model Evaluation ({pair}):")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {pair}")
    plt.show()

    # Price plot with signals
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["close"], label="Price", alpha=0.7)

    # Actual buy signals
    plt.scatter(
        data.index[data["buy_signal"] == 1],
        data["close"][data["buy_signal"] == 1],
        marker="^",
        color="g",
        label="Actual Buy",
        alpha=0.8,
    )

    # Predicted buy signals
    plt.scatter(
        data.index[data["predicted_buy_signal"] == 1],
        data["close"][data["predicted_buy_signal"] == 1],
        marker="o",
        color="r",
        label="Predicted Buy",
        alpha=0.6,
    )

    plt.legend()
    plt.title(f"Buy Signal Comparison - {pair}")
    plt.show()

    return metrics


def save_artifacts(model, metrics: dict, data: pd.DataFrame, pair: str) -> None:
    """Save model and training logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    Path("model_logs").mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = f"saved_models/buy_model_5_{pair}_{timestamp}.pkl"
    joblib.dump(model, model_path)

    # Save logs
    log_entry = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "currency_pair": pair,
                "model_path": model_path,
                **metrics,
                "num_features": len(FEATURES),
                "model_type": "LightGBM",
                "dataset_range_start": data.index.min().strftime("%Y-%m-%d"),
                "dataset_range_end": data.index.max().strftime("%Y-%m-%d"),
            }
        ]
    )

    log_path = "model_logs/train_buy_5_history.csv"
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

        # Get max F1 within 1 day window (including current model)
        recent_entries = pair_log[pd.to_datetime(pair_log["timestamp"]) >= cutoff_time]
        max_f1 = recent_entries["f1"].max() if not recent_entries.empty else 0

        for index, row in pair_log.iterrows():
            model_time = datetime.strptime(row["timestamp"], "%Y%m%d_%H%M%S")
            age_difference = current_time - model_time

            # Delete if older than 1 day OR has lower F1 than max in window
            if age_difference.days >= 1 or row["f1"] < max_f1:
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


def main():
    for _ in range(LOOP_COUNT):
        for pair in CURRENCY_PAIRS:
            # Data pipeline
            data = load_data(pair)
            data = generate_features(data)
            data = generate_labels(data)

            X = data[FEATURES]
            y = data["buy_signal"]

            # Hyperparameter optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

            # Model training
            model, _ = train_final_model(X, y, study.best_params)

            # Backtesting and saving
            metrics = backtest_model(model, data, pair)
            save_artifacts(model, metrics, data, pair)


if __name__ == "__main__":
    main()
