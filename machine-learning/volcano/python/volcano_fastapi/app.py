#!/usr/bin/env python3
"""
ML Model Server using FastAPI
Serves a simple scikit-learn model for demonstration
"""

import logging
import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    # Startup
    logger.info("Starting up ML Model Server...")
    try:
        load_model()
        logger.info("Model loaded successfully during startup")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down ML Model Server...")
    global _model
    _model = None
    logger.info("Cleanup completed")


app = FastAPI(title="ML Model Server", version="1.0.0", lifespan=lifespan)

_model: Optional[RandomForestClassifier] = None


class PredictionRequest(BaseModel):
    features: list[list[float]]


class PredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]]


def create_sample_model():
    """Create a sample model for demonstration"""
    logger.info("Creating sample Random Forest model")
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
    return model


def load_model():
    """Load model from file or create a sample one"""
    global _model

    if _model is not None:
        return _model

    model_path = os.getenv("MODEL_PATH", "/tmp/model.pkl")

    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            with open(model_path, "rb") as f:
                _model = pickle.load(f)
        else:
            logger.info("Model file not found, creating sample model")
            _model = create_sample_model()
            # Save the model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(_model, f)

        logger.info("Model loaded successfully")
        return _model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_model() -> RandomForestClassifier:
    """Dependency function to get the loaded model"""
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/health")
async def health_check(model: RandomForestClassifier = Depends(get_model)):
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None, "version": "1.0.0"}


@app.get("/ready")
async def readiness_check(model: RandomForestClassifier = Depends(get_model)):
    """Readiness check endpoint"""
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest, model: RandomForestClassifier = Depends(get_model)
):
    """Make predictions using the loaded model"""
    features = np.array(request.features)

    # Validate input shape
    if features.ndim != 2 or features.shape[1] != 20:
        raise HTTPException(
            status_code=400,
            detail=f"Expected features shape (n_samples, 20), got {features.shape}",
        )

    try:
        predictions = model.predict(features).tolist()
        probabilities = model.predict_proba(features).tolist()

        logger.info(f"Made predictions for {len(request.features)} samples")

        return PredictionResponse(predictions=predictions, probabilities=probabilities)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info(model: RandomForestClassifier = Depends(get_model)):
    """Get information about the loaded model"""
    return {
        "model_type": type(model).__name__,
        "n_features": getattr(model, "n_features_in_", "unknown"),
        "n_classes": len(getattr(model, "classes_", [])),
        "classes": getattr(model, "classes_", []).tolist(),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run("app:app", host=host, port=port, log_level="info", access_log=True)
