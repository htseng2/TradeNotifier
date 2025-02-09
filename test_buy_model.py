import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


def plot_classification(df):
    """Plot the closing prices with classified buy signals."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label="close", color="gray", alpha=0.5)

    # Plot buy signals (buy == 1) in green
    buy_signals = df[df["buy"] == 1]
    plt.scatter(buy_signals.index, buy_signals["Close"], color="green", label="Buy")

    # Plot predicted buy signals (predicted_buy == 1) in light green
    predicted_buy_signals = df[df["predicted_buy"] == 1]
    plt.scatter(
        predicted_buy_signals.index,
        predicted_buy_signals["Close"],
        color="lightgreen",
        label="Predicted Buy",
        marker="x",
    )

    plt.legend()
    plt.title("Forex Closing Prices with Buy Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


def load_model(model_path):
    """Load the LightGBM model from the specified path."""
    return lgb.Booster(model_file=model_path)


def load_data(file_path):
    """Load the data from the specified CSV file."""
    return pd.read_csv(file_path)


def predict_labels(gbm, X):
    """Predict labels using the LightGBM model."""
    y_prob = gbm.predict(X, num_iteration=gbm.best_iteration)
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    return y_pred, y_prob


def process_data(df):
    """Process the DataFrame to prepare features and true labels."""
    y_true = df["buy"]
    X = df.drop(columns=["buy", "sell"])
    return X, y_true


def main():
    parser = argparse.ArgumentParser(description="Process currency abbreviation.")
    parser.add_argument(
        "currency",
        type=str,
        nargs="?",
        default="USD",
        help="Currency abbreviation (e.g., EUR). Default is USD.",
    )
    args = parser.parse_args()

    currency = args.currency

    # Load the trained model from the models folder
    # model_path = f"models/lightgbm_model_20250209_103307.txt"
    model_path = f"models/lightgbm_model_20250209_185459.txt"
    gbm = load_model(model_path)

    # Read the DataFrame from the CSV file
    data_path = f"labeled_data/labeled_data_{currency}.csv"
    # data_path = f"labeled_data/labeled_data.csv"
    df = load_data(data_path)

    # Process data
    X, y_true = process_data(df)

    # Predict the labels and probabilities
    y_pred, y_prob = predict_labels(gbm, X)

    # Add the predicted labels to the DataFrame
    df["predicted_buy"] = y_pred

    # Save the DataFrame with the predicted labels to a new CSV file
    df.to_csv("labeled_data/labeled_data_with_predictions.csv", index=False)

    # Preview the data table head and tail
    print(df.head())
    print(df.tail())

    # Plot the data with classification
    plot_classification(df)

    # Calculate and display the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Compute ROC curve and ROC area for the binary classification
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = {0:0.2f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
