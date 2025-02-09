from matplotlib import pyplot as plt
import optuna
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
import datetime
import seaborn as sns


def prepare_features_labels(df):
    # Assuming 'label' is your target column and the rest are features
    X = df.drop(columns=["buy", "sell"])
    y = df["buy"]
    return X, y


#   Optimize hyperparameters with Optuna.
def objective(trial):
    global train_data, valid_data  # Add this line at the start of the function

    # Define the parameter search space
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "min_sum_hessian_in_leaf": trial.suggest_int("min_sum_hessian_in_leaf", 5, 20),
        "feature_pre_filter": False,  # Set this to False to avoid the error
    }

    # Train the model
    gbm = lgb.train(params, train_data, valid_sets=[valid_data])

    # Predict on the validation set
    y_pred_prob = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

    # Calculate log loss
    score = log_loss(y_valid, y_pred_prob)

    return score


def main(params):
    # Read the DataFrame from the CSV file
    df = pd.read_csv("labeled_data/labeled_data.csv")

    # Prepare features and labels
    X, y = prepare_features_labels(df)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    # Set parameters for LightGBM
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": params["num_leaves"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "min_data_in_leaf": params["min_data_in_leaf"],
        "min_sum_hessian_in_leaf": params["min_sum_hessian_in_leaf"],
        # "learning_rate": 0.1,
        # "num_leaves": 31,
        # "max_depth": 12,
        # "min_data_in_leaf": 20,
        # "feature_fraction": 0.8,
        # "bagging_fraction": 0.8,
        # "bagging_freq": 1,
        # "lambda_l2": 1.0,  # small L2 regularization
        # "verbosity": -1,  # less logging
    }

    # Train the model
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, valid_data],
        callbacks=[early_stopping(stopping_rounds=10)],
    )

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model in the models folder
    gbm.save_model(f"models/lightgbm_model_{timestamp}.txt")

    # Predict on the test set
    y_pred_prob = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

    # Convert probabilities to class labels
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]

    # Evaluate the model
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Accuracy: {accuracy}")

    # Calculate and plot the confusion matrix
    conf_matrix = confusion_matrix(y_valid, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    # Read and prepare data first
    df = pd.read_csv("labeled_data/labeled_data.csv")
    X, y = prepare_features_labels(df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create global datasets
    global train_data, valid_data
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    # Now run the optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    main(params=trial.params)
