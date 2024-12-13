import pandas as pd
import lightgbm as lgb

from data_labeler import plot_classification


def main():
    # Load the trained model
    gbm = lgb.Booster(model_file="lightgbm_model.txt")

    # Read the DataFrame from the CSV file
    df = pd.read_csv("labeled_data.csv")

    # Prepare features (assuming the label column is not present in the test data)
    X = df.drop(columns=["label"])

    # Predict the labels
    y_pred = gbm.predict(X, num_iteration=gbm.best_iteration)
    y_pred = [
        list(x).index(max(x)) for x in y_pred
    ]  # Convert probabilities to class labels

    # Add the predicted labels to the DataFrame
    df["predicted_label"] = y_pred

    # Save the DataFrame with the predicted labels to a new CSV file
    df.to_csv("labeled_data_with_predictions.csv", index=False)

    # Preview the data table head and tail
    print(df.head())
    print(df.tail())

    # Plot the data with classification
    plot_classification(df)


if __name__ == "__main__":
    main()
