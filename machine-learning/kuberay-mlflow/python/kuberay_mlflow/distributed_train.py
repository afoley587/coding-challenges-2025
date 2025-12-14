"""
RAY_API_SERVER_ADDRESS=http://localhost:8265 poetry run ray job submit -- python kuberay_mlflow/distributed_train.py
curl -X POST http://localhost:8000/fastapi_app/model/load
curl -X POST http://localhost:8000/fastapi_app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]
  }'
"""

from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import ray
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def evaluation_fn(step, width, height):
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


@ray.remote
def train_and_upload_model(
    mlflow_tracking_uri,
    width: int = 50,
    height: int = 25,
    steps: int = 100,
):
    """
    width  -> number of trees
    height -> max depth of each tree
    steps  -> dataset size + regularization proxy
    """

    formatted_date = datetime.now().strftime("%Y%d%m%H%M%S")
    experiment_name = f"simple_model_training_{formatted_date}"

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    n_estimators = max(10, width)
    max_depth = max(2, height)
    max_features = min(1.0, 0.3 + (steps / 500))

    n_samples = min(5000, 500 + steps * 10)
    n_features = 20
    n_informative = max(2, n_features // 2)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "width": width,
                "height": height,
                "steps": steps,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": max_features,
                "n_samples": n_samples,
                "model_type": "RandomForestClassifier",
            }
        )

        print(
            f"Training RandomForest("
            f"n_estimators={n_estimators}, "
            f"max_depth={max_depth}, "
            f"max_features={max_features})"
        )

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            n_classes=2,
            random_state=42,
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=1,
            random_state=42,
        )

        model.fit(X, y)

        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "final_loss": evaluation_fn(steps - 1, width, height),
            }
        )

        mlflow.sklearn.log_model(model, "random_forest_model")

        result = mlflow.register_model(
            f"runs:/{run.info.run_id}/random_forest_model",
            "RandomForestClassifierModel",
        )

        return {
            "accuracy": accuracy,
            "model_source": result.source,
            "run_id": run.info.run_id,
        }


mlflow_tracking_uri = "http://mlflow:80"

future = train_and_upload_model.remote(
    mlflow_tracking_uri, width=75, height=30, steps=200
)

result = ray.get(future)
print(f"Training completed: {result}")
