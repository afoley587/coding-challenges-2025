import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import ollama
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from ollama import ResponseError
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    gc_collector,
    generate_latest,
    platform_collector,
    process_collector,
)
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
class Config:
    """Centralized application configuration"""

    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://ollama-service:11434")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# Prometheus Metrics
class Metrics:
    """Centralized Prometheus metrics"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Register default collectors
        gc_collector.GCCollector(registry=self.registry)
        platform_collector.PlatformCollector(registry=self.registry)
        process_collector.ProcessCollector(registry=self.registry)

        # Custom metrics
        self.request_count = Counter(
            "afoley_vllm_requests_total",
            "Total LLM requests",
            ["status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "afoley_vllm_request_duration_seconds",
            "Request duration in seconds",
            registry=self.registry,
        )

        self.tokens_generated = Counter(
            "afoley_vllm_tokens_generated_total",
            "Total tokens generated",
            registry=self.registry,
        )


class AppState:
    """Application state container"""

    config: Config
    metrics: Metrics
    ollama_client: ollama.Client


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events
    """
    logger.info("Starting up application...")

    state.config = Config()
    logger.info(
        f"Configuration loaded: OLLAMA_HOST={state.config.OLLAMA_HOST}, MODEL={state.config.MODEL_NAME}"
    )

    state.metrics = Metrics()
    logger.info("Prometheus metrics initialized")

    state.ollama_client = ollama.Client(host=state.config.OLLAMA_HOST)
    logger.info(f"Ollama client initialized for {state.config.OLLAMA_HOST}")

    try:
        models = state.ollama_client.list()
        logger.info(
            f"Connected to Ollama. Available models: {[m['model'] for m in models.get('models', [])]}"
        )
    except Exception as e:
        logger.warning(f"Could not connect to Ollama on startup: {e}")

    logger.info("Application startup complete")

    yield

    logger.info("Shutting down application...")
    logger.info("Application shutdown complete")


def get_config() -> Config:
    """Dependency for accessing configuration"""
    return state.config


def get_metrics() -> Metrics:
    """Dependency for accessing metrics"""
    return state.metrics


def get_ollama_client() -> ollama.Client:
    """Dependency for accessing Ollama client"""
    return state.ollama_client


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate from")
    max_tokens: Optional[int] = Field(
        256, ge=1, le=2048, description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold"
    )


class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    tokens_generated: int
    finish_reason: str
    model: str


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    ollama: str
    model_loaded: bool
    model_name: str


app = FastAPI(
    title="LLM API Gateway",
    description="API gateway for Ollama inference",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root(config: Config = Depends(get_config)):
    """Root endpoint with basic service info"""
    return {
        "status": "healthy",
        "service": "llm-api-gateway",
        "ollama_host": config.OLLAMA_HOST,
        "model": config.MODEL_NAME,
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Liveness probe - checks if the service is alive

    This endpoint just performs a basic `ping` check
    """
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadinessResponse)
async def readiness(
    client: ollama.Client = Depends(get_ollama_client),
    config: Config = Depends(get_config),
):
    """
    Readiness probe - checks if the service is ready to accept traffic

    This endpoint verifies that:
    1. Ollama is reachable
    2. The configured model is loaded and available
    """
    try:
        models = client.list()
        model_names = {model["model"] for model in models.get("models", [])}
        model_loaded = config.MODEL_NAME in model_names

        if not model_loaded:
            logger.warning(
                f"Model {config.MODEL_NAME} not found. Available: {model_names}"
            )
            raise HTTPException(
                status_code=503,
                detail=f"Model {config.MODEL_NAME} not loaded. Available: {model_names}",
            )

        return ReadinessResponse(
            status="ready",
            ollama="connected",
            model_loaded=True,
            model_name=config.MODEL_NAME,
        )

    except HTTPException:
        raise
    except ResponseError as e:
        logger.error(f"Ollama error during readiness check: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@app.get("/metrics")
async def get_metrics_endpoint(metrics: Metrics = Depends(get_metrics)):
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST
    )


@app.post("/v1/generate", response_model=GenerationResponse)
async def generate(
    request: GenerationRequest,
    client: ollama.Client = Depends(get_ollama_client),
    config: Config = Depends(get_config),
    metrics: Metrics = Depends(get_metrics),
):
    """
    Generate text using Ollama

    Ollama provides efficient local inference optimized for Apple Silicon.
    """
    start_time = time.time()

    try:
        logger.info(f"Sending to Ollama: prompt_length={len(request.prompt)}")

        response = client.generate(
            model=config.MODEL_NAME,
            prompt=request.prompt,
            options={
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        )

        generated_text = response.get("response", "")
        tokens_generated_count = response.get("eval_count", 0)
        finish_reason = "stop" if response.get("done", False) else "length"

        metrics.request_count.labels(status="success").inc()
        metrics.request_duration.observe(time.time() - start_time)
        metrics.tokens_generated.inc(tokens_generated_count)

        duration = time.time() - start_time
        logger.info(f"Generated {tokens_generated_count} tokens in {duration:.2f}s")

        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=tokens_generated_count,
            finish_reason=finish_reason,
            model=response.get("model", config.MODEL_NAME),
        )

    except ResponseError as e:
        metrics.request_count.labels(status="error").inc()
        logger.error(f"Ollama error: {e}")

        status_code = getattr(e, "status_code", 500)
        raise HTTPException(status_code=status_code, detail=f"Ollama error: {str(e)}")

    except TimeoutError:
        metrics.request_count.labels(status="timeout").inc()
        logger.error("Request timed out")
        raise HTTPException(status_code=504, detail="Request timed out")

    except Exception as e:
        metrics.request_count.labels(status="error").inc()
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/v1/models")
async def list_models(client: ollama.Client = Depends(get_ollama_client)):
    """List available models from Ollama"""
    try:
        return client.list()
    except ResponseError as e:
        logger.error(f"Ollama error listing models: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=503, detail="Couldn't get models from Ollama")


if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True,
    )
