# import mlflow
# from mlflow.models import infer_signature

# import pandas as pd
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # Load the Iris dataset
# X, y = datasets.load_iris(return_X_y=True)

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Define the model hyperparameters
# params = {
#     "solver": "lbfgs",
#     "max_iter": 1000,
#     "multi_class": "auto",
#     "random_state": 8888,
# }

# # Train the model
# lr = LogisticRegression(**params)
# lr.fit(X_train, y_train)

# # Predict on the test set
# y_pred = lr.predict(X_test)

# # Calculate metrics
# accuracy = accuracy_score(y_test, y_pred)
import os
import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up the MLflow tracking URI to a local directory
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
os.environ["MLFLOW_ARTIFACT_ROOT"] = "./mlruns"

mlflow.set_tracking_uri("file:./mlruns")

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)
feature_names = datasets.load_iris().feature_names

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log the metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log the model
    mlflow.sklearn.log_model(lr, "model")

    # Log feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(lr.coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    # Log predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    predictions_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv")

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

print("MLflow run completed. Check the MLflow UI for detailed results.")