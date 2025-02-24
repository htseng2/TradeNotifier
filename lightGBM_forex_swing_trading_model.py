import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
import talib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import joblib

currency_pairs = [
    "USD_TWD",
    "EUR_TWD",
    "GBP_TWD",
    "AUD_TWD",
    "CHF_TWD",
    "NZD_TWD",
    "JPY_TWD",
]

loop_count = 10


def main():
    for i in range(loop_count):
        for pair in currency_pairs:
            # 1️⃣ Load Your Forex Data
            data = pd.read_csv(
                f"Alpha_Vantage_Data/{pair}.csv",
                skiprows=1,  # Skip malformed header row
                header=None,
                names=["timestamp", "open", "high", "low", "close"],
                parse_dates=["timestamp"],
            )
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)

            # Filter to last 10 years ⬇️
            data = data.loc[data.index >= pd.Timestamp.now() - pd.DateOffset(years=10)]
            print(data.head())

            # 2️⃣ Feature Engineering: Common Swing Trading Indicators
            def generate_features(df):
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
                # df["Volume_change"] = df["volume"].pct_change()
                df.dropna(inplace=True)  # Remove NaN values
                return df

            data = generate_features(data)

            # 3️⃣ Labeling: Buy Signals Based on Trend
            def generate_labels(df):
                df["future_return"] = (
                    df["close"].shift(-5) / df["close"] - 1
                )  # Predict 5 steps ahead
                df["buy_signal"] = (df["future_return"] > 0.002).astype(
                    int
                )  # Buy if expected return > 0.2%
                df.drop(columns=["future_return"], inplace=True)
                return df

            data = generate_labels(data)

            # 4️⃣ Define Training & Validation Data
            features = [
                "SMA_50",
                "SMA_200",
                "RSI",
                "MACD",
                "BB_upper",
                "BB_lower",
                "ATR",
                # "Volume_change",
            ]
            target = "buy_signal"

            X = data[features]
            y = data[target]

            # 5️⃣ Time Series Split for Validation
            tscv = TimeSeriesSplit(n_splits=5)

            # 6️⃣ Hyperparameter Optimization with Optuna
            def objective(trial):
                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "verbosity": -1,
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                    "feature_fraction": trial.suggest_float(
                        "feature_fraction", 0.5, 1.0
                    ),
                    "bagging_fraction": trial.suggest_float(
                        "bagging_fraction", 0.5, 1.0
                    ),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
                }

                # Track multiple metrics
                metrics = {
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "f1": [],
                    "roc_auc": [],
                }

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

                    # Calculate all metrics
                    metrics["accuracy"].append(accuracy_score(y_val, y_pred))
                    metrics["precision"].append(precision_score(y_val, y_pred))
                    metrics["recall"].append(recall_score(y_val, y_pred))
                    metrics["f1"].append(f1_score(y_val, y_pred))
                    metrics["roc_auc"].append(roc_auc_score(y_val, y_pred))

                # Print trial metrics
                print(f"\nTrial {trial.number} Metrics:")
                for metric, values in metrics.items():
                    print(f"{metric.capitalize()}: {np.mean(values):.4f}")

                return np.mean(metrics["accuracy"])  # Still optimize for accuracy

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)
            best_params = study.best_params

            # 7️⃣ Train Final LightGBM Model
            final_model = lgb.LGBMClassifier(**best_params)
            final_model.fit(X, y)

            # 7️⃣ Train and Evaluate Final Model
            print("\nFinal Model Evaluation:")
            final_predictions = final_model.predict(X)

            # Store metrics for logging
            metrics = {
                "accuracy": accuracy_score(y, final_predictions),
                "precision": precision_score(y, final_predictions),
                "recall": recall_score(y, final_predictions),
                "f1": f1_score(y, final_predictions),
                "roc_auc": roc_auc_score(y, final_predictions),
            }

            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")

            # 8️⃣ Generate Buy Signals for Trading
            data["predicted_buy_signal"] = final_model.predict(X)

            # # 🔼 Add this new section for visualization
            # plt.figure(figsize=(15, 8))
            # plt.subplot(2, 1, 1)
            # plt.plot(data.index, data["close"], label="Price", alpha=0.5)
            # plt.plot(
            #     data.index[data["buy_signal"] == 1],
            #     data["close"][data["buy_signal"] == 1],
            #     "^",
            #     markersize=10,
            #     color="g",
            #     label="Actual Buy Signals",
            # )
            # plt.plot(
            #     data.index[data["predicted_buy_signal"] == 1],
            #     data["close"][data["predicted_buy_signal"] == 1],
            #     "o",
            #     markersize=8,
            #     color="r",
            #     alpha=0.7,
            #     label="Predicted Buy Signals",
            # )
            # plt.title("Buy Signal Comparison")
            # plt.ylabel("Price")
            # plt.legend()

            # plt.subplot(2, 1, 2)
            # plt.plot(data.index, data["RSI"], label="RSI", color="purple", alpha=0.7)
            # plt.axhline(70, linestyle="--", color="r", alpha=0.5)
            # plt.axhline(30, linestyle="--", color="g", alpha=0.5)
            # plt.ylabel("RSI")
            # plt.xlabel("Date")

            # plt.tight_layout()
            # plt.savefig("signal_comparison.png")
            # plt.show()

            # 9️⃣ Save Model and Log Training Session
            # Create models directory if not exists
            Path("saved_models").mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"saved_models/lightgbm_forex_swing_model_{timestamp}.pkl"
            joblib.dump(final_model, model_filename)
            print(f"✅ Model saved to: {model_filename}")

            # Create model logs directory if not exists
            Path("model_logs").mkdir(parents=True, exist_ok=True)

            # Create training log entry
            log_entry = {
                "timestamp": timestamp,
                "currency_pair": pair,
                "model_path": model_filename,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "num_features": len(features),
                "model_type": "LightGBM",
                "dataset_range_start": data.index.min().strftime("%Y-%m-%d"),
                "dataset_range_end": data.index.max().strftime("%Y-%m-%d"),
            }

            # Append to training history CSV
            log_df = pd.DataFrame([log_entry])
            log_path = "model_logs/training_history.csv"

            if Path(log_path).exists():
                log_df.to_csv(log_path, mode="a", header=False, index=False)
            else:
                log_df.to_csv(log_path, index=False)

            print(f"✅ Training log saved to: {log_path}")

            print("✅ Model Training Complete. Ready for Backtesting!")


if __name__ == "__main__":
    main()
