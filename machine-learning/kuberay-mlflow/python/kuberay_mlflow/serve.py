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

        # Set MLflow tracking URI
        self.mlflow_tracking_uri = mlflow_uri
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    def load_model(self):
        """Load model from MLflow model registry"""
        try:
            logger.info(f"Loading model {self.model_name} version {self.model_version}")

            # Load model from MLflow
            if self.model_version == "latest":
                model_uri = f"models:/{self.model_name}/Latest"
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"

            self.model = mlflow.sklearn.load_model(model_uri)

            # Get model metadata
            client = mlflow.tracking.MlflowClient()
            try:
                if self.model_version == "latest":
                    model_version = client.get_latest_versions(
                        self.model_name, stages=["None", "Staging", "Production"]
                    )[0]
                else:
                    model_version = client.get_model_version(
                        self.model_name, self.model_version
                    )

                self.model_info = {
                    "name": self.model_name,
                    "version": model_version.version,
                    "stage": model_version.current_stage,
                    "description": model_version.description
                    or "No description available",
                    "creation_timestamp": model_version.creation_timestamp,
                }
            except Exception as e:
                logger.warning(f"Could not load model metadata: {e}")
                self.model_info = {
                    "name": self.model_name,
                    "version": self.model_version,
                    "stage": "Unknown",
                    "description": "Loaded directly from URI",
                    "creation_timestamp": None,
                }

            logger.info(f"Model loaded successfully: {self.model_info}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create a dummy model for demo purposes
            raise (e)

    @app.get("/")
    def root(self):
        """Root endpoint"""
        return {
            "message": "ML Model Server is running",
            "model": self.model_info.get("name", "Unknown"),
            "version": self.model_info.get("version", "Unknown"),
            "timestamp": datetime.now().isoformat(),
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(self, request: PredictionRequest):
        """Make predictions on input data"""
        try:
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            # Convert input to numpy array
            X = np.array(request.features)

            # Validate input shape
            if len(X.shape) != 2:
                raise HTTPException(
                    status_code=400, detail=f"Expected 2D array, got shape {X.shape}"
                )

            # Make predictions
            predictions = self.model.predict(X).tolist()

            # Get prediction probabilities if available
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probabilities = self.model.predict_proba(X).tolist()
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")

            return PredictionResponse(
                predictions=predictions,
                probabilities=probabilities,
                model_version=self.model_info.get("version", "unknown"),
                timestamp=datetime.now().isoformat(),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.get("/health", response_model=HealthResponse)
    def health(self):
        """Health check endpoint"""
        try:
            # Get Ray cluster information
            ray_info = {
                "cluster_resources": ray.cluster_resources(),
                "available_resources": ray.available_resources(),
                "nodes": len(ray.nodes()),
            }
        except Exception as e:
            logger.warning(f"Could not get Ray cluster info: {e}")
            ray_info = {"error": str(e)}

        return HealthResponse(
            status="healthy" if self.model is not None else "unhealthy",
            model_loaded=self.model is not None,
            model_info=self.model_info,
            ray_cluster_info=ray_info,
            timestamp=datetime.now().isoformat(),
        )

    @app.get("/model/load")
    def load(self):
        """Load model and track with MLFlow"""
        self.load_model()
        return {
            "model_info": self.model_info,
            "mlflow_uri": self.mlflow_tracking_uri,
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/model/info")
    def model_info_endpoint(self):
        """Get model information"""
        return {
            "model_info": self.model_info,
            "mlflow_uri": self.mlflow_tracking_uri,
            "timestamp": datetime.now().isoformat(),
        }


deployment = ModelServer.bind(
    model_name="RandomForestClassifierModel",
    model_version="latest",
    mlflow_uri="http://mlflow:80",
)
