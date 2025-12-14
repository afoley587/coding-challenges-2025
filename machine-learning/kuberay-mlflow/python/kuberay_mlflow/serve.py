"""
Model Serving Module

This module provides model serving capabilities using Ray Serve
with MLflow model loading and FastAPI endpoints following the official Ray Serve pattern.
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    features: list[list[float]]
    feature_names: Optional[list[str]] = None


class PredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: Optional[list[list[float]]] = None
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: dict[str, Any]
    ray_cluster_info: dict[str, Any]
    timestamp: str


class RootResponse(BaseModel):
    message: str
    model_name: str
    model_version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_info: dict[str, Any]
    mlflow_uri: str
    timestamp: str


app = FastAPI(
    title="ML Model Server",
    description="Machine Learning Model Serving API using Ray Serve and MLflow",
    version="1.0.0",
)


@serve.deployment
@serve.ingress(app)
class ModelServer:
    """Ray Serve deployment for ML model serving with FastAPI integration"""

    def __init__(
        self,
        model_name: str = "RandomForestClassifierModel",
        model_version: str = "latest",
        mlflow_uri: str = "http://mlflow-service:5000",
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.model_info = {}

        self.mlflow_tracking_uri = mlflow_uri
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    def load_model(self) -> None:
        """Load model from MLflow model registry"""
        logger.info("Attempting to load model")

        if not self.model_name:
            raise ValueError("Model name must be provided")

        try:
            if self.model_version == "latest":
                model_uri = f"models:/{self.model_name}/Latest"
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"

            self.model = mlflow.sklearn.load_model(model_uri)

            client = mlflow.tracking.MlflowClient()

            if self.model_version == "latest":
                versions = client.get_latest_versions(
                    self.model_name, stages=["None", "Staging", "Production"]
                )
                if not versions:
                    raise LookupError("No registered versions found for model")
                model_version = versions[0]
            else:
                model_version = client.get_model_version(
                    self.model_name, self.model_version
                )

            self.model_info = {
                "name": self.model_name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "description": model_version.description or "No description available",
                "creation_timestamp": model_version.creation_timestamp,
            }

            logger.info("Model loaded successfully")

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"MLflow error while loading model: {e}")
            raise

        except Exception as e:
            logger.exception("Unexpected error while loading model")
            raise

    @app.get("/", response_model=RootResponse)
    def root(self):
        """Root endpoint"""
        return RootResponse(
            message="ML Model Server is running",
            model_name=self.model_info.get("name", "unknown"),
            model_version=self.model_info.get("version", "unknown"),
            timestamp=datetime.now().isoformat(),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(self, request: PredictionRequest):
        """Make predictions on input data"""
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Call /model/load first.",
            )

        try:
            X = np.asarray(request.features)

            if X.ndim != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected 2D array [n_samples, n_features], got shape {X.shape}",
                )

            if X.size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Input features array is empty",
                )

            predictions = self.model.predict(X).tolist()

            probabilities = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X).tolist()

            return PredictionResponse(
                predictions=predictions,
                probabilities=probabilities,
                model_version=self.model_info.get("version", "unknown"),
                timestamp=datetime.now().isoformat(),
            )

        except HTTPException:
            raise

        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input values: {str(e)}",
            )

        except Exception as e:
            logger.exception("Unhandled prediction error")
            raise HTTPException(
                status_code=500,
                detail="Internal prediction error",
            )

    @app.get("/health", response_model=HealthResponse)
    def health(self):
        """Health check endpoint"""
        model_loaded = self.model is not None
        ray_info: dict[str, Any]

        try:
            ray_info = {
                "cluster_resources": ray.cluster_resources(),
                "available_resources": ray.available_resources(),
                "nodes": len(ray.nodes()),
            }
        except Exception as e:
            logger.warning(f"Ray health check failed: {e}")
            ray_info = {
                "status": "unavailable",
                "error": str(e),
            }

        status = "healthy" if model_loaded else "degraded"

        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            model_info=self.model_info,
            ray_cluster_info=ray_info,
            timestamp=datetime.now().isoformat(),
        )

    @app.post("/model/load", response_model=ModelInfoResponse)
    def load(self):
        """Load model from MLflow into memory"""
        try:
            self.load_model()
            return ModelInfoResponse(
                model_loaded=True,
                model_info=self.model_info,
                mlflow_uri=self.mlflow_tracking_uri,
                timestamp=datetime.now().isoformat(),
            )

        except LookupError as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {str(e)}",
            )

        except mlflow.exceptions.MlflowException as e:
            raise HTTPException(
                status_code=502,
                detail=f"MLflow error while loading model: {str(e)}",
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}",
            )

    @app.get("/model/info")
    def model_info_endpoint(self):
        """Get model information"""
        return ModelInfoResponse(
            model_loaded=self.model is not None,
            model_info=self.model_info,
            mlflow_uri=self.mlflow_tracking_uri,
            timestamp=datetime.now().isoformat(),
        )


deployment = ModelServer.bind(
    model_name="RandomForestClassifierModel",
    model_version="latest",
    mlflow_uri="http://mlflow:80",
)
