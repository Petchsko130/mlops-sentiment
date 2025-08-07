import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 200,
    "multi_class": "auto",
    "random_state": 8888,
}

model = LogisticRegression(**params)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Before you run this check your MLFLOW Tracking Server Status before
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
mlflow.set_experiment("MLflow MLOps-Sentiment")

# Start MLflow run
with mlflow.start_run() as run:
    params["features_used"] = X.columns

    # Without Autologging
    # Log parameters and metrics
    mlflow.log_param("Hyperparameters", params)
    mlflow.log_metric("accuracy", acc)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # ✅ Log the model
    print("📦 Logging model...")
    mlflow.sklearn.log_model(
        sk_model=model,
        name="lr_model",
        input_example=X_train.head(),
        signature=mlflow.models.signature.infer_signature(
            X_train, model.predict(X_train)
        ),
    )
    print("✅ Logged!")

    print("✅ Model trained and logged.")
    print("🔗 Run ID:", run.info.run_id)
