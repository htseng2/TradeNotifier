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
    return pd.read_csv(file_path)


def prepare_data_table(df):
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df.set_index("Date", inplace=True)
    df = df.sort_index()
    df["Close"] = df["Close"].astype(float)
    return df[df.index.dayofweek < 5]


def add_technical_indicators(df):
    # Moving Averages
    for window in [10, 50, 200]:
        df[f"MA_{window}"] = df["Close"].rolling(window).mean()

    # Price Extremes
    for window in [10, 21, 50, 100, 200]:
        df[f"Max_{window}"] = df["Close"].rolling(window).max()
        df[f"Min_{window}"] = df["Close"].rolling(window).min()

    # Momentum Indicators
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    return df


def generate_labels(df, lookahead=21, expected_return=0.005, spread=0.02):
    df["buy"] = 0
    threshold = 1 + expected_return + spread

    for i in range(len(df) - lookahead):
        current_price = df["Close"].iloc[i]
        future_prices = df["Close"].iloc[i + 1 : i + lookahead + 1]

        if any(future_prices > current_price * threshold):
            df.at[df.index[i], "buy"] = 1

    return df.iloc[lookahead:-lookahead]


# ----------------------------
# Model Training Functions
# ----------------------------
def prepare_features(df):
    return df.drop(columns=["buy"]), df["buy"]


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
    currency_pairs = [
        ("USD", "TWD"),
        ("CNY", "TWD"),
        ("EUR", "TWD"),
        ("SGD", "TWD"),
        ("GBP", "TWD"),
        ("AUD", "TWD"),
        ("CHF", "TWD"),
        ("CAD", "TWD"),
        ("JPY", "TWD"),
        ("HKD", "TWD"),
    ]
    dfs = []

    for pair in currency_pairs:
        df = fetch_forex_data(f"Alpha_Vantage_Data/{pair[0]}_{pair[1]}.csv")
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

    # Feature Importance Analysis
    print("\n=== Feature Importance ===")
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(
        model.booster_,
        max_num_features=20,
        importance_type="gain",
        title="Feature Importance (Gain)",
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

    # Save results to history
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
    }

    save_results_report(
        results_path="lightGBM_model_history.csv",
        metrics=metrics,
        features=list(X.columns),
        params=best_params,
    )

    # Save Model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.booster_.save_model(f"models/buy_model_{timestamp}.txt")


if __name__ == "__main__":
    main()
