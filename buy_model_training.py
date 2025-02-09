from matplotlib import pyplot as plt
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import datetime
import seaborn as sns


def prepare_features_labels(df):
    # Assuming 'label' is your target column and the rest are features
    X = df.drop(columns=["buy", "sell"])
    y = df["buy"]
    return X, y


def main():
    # Read the DataFrame from the CSV file
    df = pd.read_csv("labeled_data/labeled_data.csv")

    # Prepare features and labels
    X, y = prepare_features_labels(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Set parameters for LightGBM
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 12,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,  # small L2 regularization
        "verbosity": -1,  # less logging
    }

    # Train the model
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        callbacks=[early_stopping(stopping_rounds=10)],
    )

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model in the models folder
    gbm.save_model(f"models/lightgbm_model_{timestamp}.txt")

    # Predict on the test set
    y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # Convert probabilities to class labels
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Calculate and plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    main()
