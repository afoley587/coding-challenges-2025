"""
RAY_API_SERVER_ADDRESS=http://localhost:8265 poetry run ray job submit -- python kuberay_mlflow/distributed_train.py
curl http://localhost:8000/fastapi_app/model/load
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


def evaluation_fn(step, width, height):
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


@ray.remote
def train_and_upload_model(mlflow_tracking_uri, width=50, height=25, steps=100):
    formatted_date = datetime.now().strftime("%Y%d%m%H%M%S")
    experiment_name = f"simple_model_training_{formatted_date}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:

        mlflow.log_params(
            {
                "width": width,
                "height": height,
                "steps": steps,
                "model_type": "RandomForestClassifier",
            }
        )

        print(f"Training model with width={width}, height={height}, steps={steps}")

        # Generate some random training data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=10,
            n_classes=2,
            random_state=42,
        )

        model = RandomForestClassifier(
            n_estimators=10, random_state=42, n_jobs=1  # Small for demo
        )
        model.fit(X, y)

        # Calculate final metrics
        predictions = model.predict(X)
        mse = np.mean((y - predictions) ** 2)

        # Log metrics
        mlflow.log_metrics(
            {"mse": mse, "final_loss": evaluation_fn(steps - 1, width, height)}
        )

        mlflow.sklearn.log_model(
            model,
            "random_forest_model",
        )
        result = mlflow.register_model(
            f"runs:/{run.info.run_id}/random_forest_model",
            "RandomForestClassifierModel",
        )

        return {"mse": mse, "model_source": result.source}


mlflow_tracking_uri = "http://mlflow:80"

future = train_and_upload_model.remote(
    mlflow_tracking_uri, width=75, height=30, steps=200
)

result = ray.get(future)
print(f"Training completed: {result}")
