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
    "EUR_TWD",
    "GBP_TWD",
    "AUD_TWD",
    "CHF_TWD",
    "NZD_TWD",
    "JPY_TWD",
]
LOOP_COUNT = 10
TRAINING_DATA_YEARS_MAX = 10
TRAINING_DATA_YEARS_MIN = 3
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
        df.index >= pd.Timestamp.now() - pd.DateOffset(years=TRAINING_DATA_YEARS_MAX)
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


def find_gross_expected_return(
    df: pd.DataFrame,
    target_percentage: float = 25.0,
    lookahead: int = 21,
    tolerance: float = 0.5,
    max_iterations: int = 100,
    verbose: bool = True,
) -> float:
    """
    Binary search for optimal return threshold that produces target buy percentage.

    Args:
        df: DataFrame containing price data
        target_percentage: Desired percentage of buy signals
        lookahead: Lookahead window size for label generation
        tolerance: Acceptable percentage deviation from target
        max_iterations: Maximum search iterations
        verbose: Whether to print search progress

    Returns:
        Optimal gross expected return value
    """
    lower, upper = 0.0, 0.05
    optimal_expected_return = None

    def calculate_buy_percentage(threshold: float) -> float:
        """Helper function to calculate buy percentage for given threshold"""
        temp_df = generate_labels(df.copy(), lookahead, threshold)
        return (temp_df["buy_signal"].sum() / len(temp_df)) * 100

    for iteration in range(max_iterations):
        current_threshold = (lower + upper) / 2
        buy_percentage = calculate_buy_percentage(current_threshold)

        if verbose:
            print(
                f"Iteration {iteration + 1}: Threshold={current_threshold:.5f}, "
                f"Buy%={buy_percentage:.2f}"
            )

        if abs(buy_percentage - target_percentage) <= tolerance:
            optimal_expected_return = current_threshold
            break

        # Adjust search boundaries
        if buy_percentage < target_percentage:
            upper = current_threshold  # Need more buys, lower threshold
        else:
            lower = current_threshold  # Need fewer buys, raise threshold

    if optimal_expected_return is None:
        optimal_expected_return = current_threshold
        if verbose:
            print("Warning: Optimal threshold not found within max iterations")

    if verbose:
        print(
            f"Final threshold: {optimal_expected_return:.5f} "
            f"(Buy%: {buy_percentage:.2f})"
        )

    return optimal_expected_return


def generate_labels(
    df: pd.DataFrame,
    lookahead: int = 20,
    gross_expected_return: float = 0.025,
) -> pd.DataFrame:
    """Create binary buy signals based on future returns"""
    # Calculate forward-looking maximum price
    reversed_close = df["close"][::-1]
    future_max = reversed_close.rolling(lookahead, min_periods=1).max()[::-1].shift(-1)
    df["buy_signal"] = (future_max > df["close"] * (1 + gross_expected_return)).astype(
        int
    )

    # Clean up edge cases
    df["buy_signal"] = df["buy_signal"].fillna(0).astype(int)
    return df


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

    # # Plot confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.title(f"Confusion Matrix - {pair}")
    # plt.show()

    # # Price plot with signals
    # plt.figure(figsize=(12, 6))
    # plt.plot(data.index, data["close"], label="Price", alpha=0.7)

    # # Actual buy signals
    # plt.scatter(
    #     data.index[data["buy_signal"] == 1],
    #     data["close"][data["buy_signal"] == 1],
    #     marker="^",
    #     color="g",
    #     label="Actual Buy",
    #     alpha=0.8,
    # )

    # # Predicted buy signals
    # plt.scatter(
    #     data.index[data["predicted_buy_signal"] == 1],
    #     data["close"][data["predicted_buy_signal"] == 1],
    #     marker="o",
    #     color="r",
    #     label="Predicted Buy",
    #     alpha=0.6,
    # )

    # plt.legend()
    # plt.title(f"Buy Signal Comparison - {pair}")
    # plt.show()

    return metrics


def save_artifacts(
    model, metrics: dict, data: pd.DataFrame, pair: str, gross_er: float
) -> None:
    """Save model and training logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    Path("model_logs").mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = f"saved_models/buy_model_20_{pair}_{timestamp}.pkl"
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
                "num_features": len(FEATURES),
                "model_type": "LightGBM",
                "dataset_range_start": data.index.min().strftime("%Y-%m-%d"),
                "dataset_range_end": data.index.max().strftime("%Y-%m-%d"),
            }
        ]
    )

    log_path = "model_logs/train_buy_20_history.csv"
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


def main():
    for pair in CURRENCY_PAIRS.copy():
        current_training_years = TRAINING_DATA_YEARS_MAX
        model_found = False

        while current_training_years >= TRAINING_DATA_YEARS_MIN:
            if Path("model_logs/train_buy_20_history.csv").exists():
                log_df = pd.read_csv("model_logs/train_buy_20_history.csv")
                recent_models = log_df[
                    (log_df["currency_pair"] == pair)
                    & (
                        pd.to_datetime(log_df["timestamp"], format="%Y%m%d_%H%M%S")
                        > pd.Timestamp.now() - pd.Timedelta(days=1)
                    )
                    & (log_df["f1"] > 0.9)
                    & (log_df["precision"] < 1.0)
                ]
                if not recent_models.empty:
                    print(f"🚨 Skipping {pair} - recent model with F1 > 0.9 exists")
                    CURRENCY_PAIRS.remove(pair)
                    model_found = True
                    break

            for _ in range(LOOP_COUNT):
                data = load_data(pair).loc[
                    lambda df: df.index
                    >= pd.Timestamp.now() - pd.DateOffset(years=current_training_years)
                ]
                data = generate_features(data)
                gross_expected_return = find_gross_expected_return(data)
                data = generate_labels(
                    data, gross_expected_return=gross_expected_return
                )

                X = data[FEATURES]
                y = data["buy_signal"]

                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

                model, _ = train_final_model(X, y, study.best_params)

                metrics = backtest_model(model, data, pair)
                save_artifacts(model, metrics, data, pair, gross_expected_return)

                if metrics.get("f1", 0) > 0.9 and metrics.get("precision", 1.0) < 1.0:
                    print(
                        f"✅ Successfully trained model for {pair} with F1 > 0.9 and precision < 1.0"
                    )
                    CURRENCY_PAIRS.remove(pair)
                    model_found = True
                    break

            if model_found:
                break

            current_training_years -= 1
            print(
                f"🔁 Reducing training window to {current_training_years} years for {pair}"
            )

        if not model_found:
            print(f"❌ Giving up on {pair} - minimum training window reached")
            CURRENCY_PAIRS.remove(pair)

        if not CURRENCY_PAIRS:
            print("🎉 All currency pairs processed")
            break


if __name__ == "__main__":
    main()
