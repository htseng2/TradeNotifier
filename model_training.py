import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def prepare_features_labels(df):
    # Assuming 'label' is your target column and the rest are features
    X = df.drop(columns=["label"])
    y = df["label"]
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
        "objective": "multiclass",
        "num_class": 3,  # Number of classes
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }

    # Train the model
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        callbacks=[early_stopping(stopping_rounds=10)],
    )

    # Save the model
    gbm.save_model("lightgbm_model.txt")

    # Predict on the test set
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred = [
        list(x).index(max(x)) for x in y_pred
    ]  # Convert probabilities to class labels

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
