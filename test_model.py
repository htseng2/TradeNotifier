import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize


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

    # Plot predicted buy signals (predicted_label == 0) in light green
    predicted_buy_signals = df[df["predicted_label"] == 0]
    plt.scatter(
        predicted_buy_signals.index,
        predicted_buy_signals["Close"],
        color="lightgreen",
        label="Predicted Buy",
        marker="x",
    )

    # Plot predicted sell signals (predicted_label == 2) in light red
    predicted_sell_signals = df[df["predicted_label"] == 2]
    plt.scatter(
        predicted_sell_signals.index,
        predicted_sell_signals["Close"],
        color="lightcoral",
        label="Predicted Sell",
        marker="x",
    )

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
    gbm = lgb.Booster(model_file=f"models/lightgbm_model_20250209_153949.txt")

    # Read the DataFrame from the CSV file
    # df = pd.read_csv(f"labeled_data/labeled_data_{currency}.csv")
    df = pd.read_csv("labeled_data/labeled_data.csv")

    # Get the true labels before dropping the label column
    y_true = df["label"]

    # Prepare features (assuming the label column is not present in the test data)
    X = df.drop(columns=["label"])

    # Predict the labels
    y_prob = gbm.predict(X, num_iteration=gbm.best_iteration)
    y_pred = [
        list(x).index(max(x)) for x in y_prob
    ]  # Convert probabilities to class labels

    # Add the predicted labels to the DataFrame
    df["predicted_label"] = y_pred

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

    # Binarize the output for multi-class ROC-AUC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])  # Adjust classes as needed
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], [prob[i] for prob in y_prob])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], [prob[i] for prob in y_prob])

    # Plot ROC curve for each class
    plt.figure()
    colors = ["aqua", "darkorange", "cornflowerblue"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic for Multi-Class")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
