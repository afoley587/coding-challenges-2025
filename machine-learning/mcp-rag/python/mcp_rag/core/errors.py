"""Custom exceptions and error helpers for the MCP RAG service."""

from __future__ import annotations

from typing import Any, Optional

from fastapi.responses import JSONResponse
from mcp_rag.schemas import RAGErrorResponse


class RAGServiceError(Exception):
    """Raised when a RAG operation fails in a recoverable way."""

    def __init__(self, message: str, *, context: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class ConfigurationError(RuntimeError):
    """Raised when application configuration is invalid."""


class DependencyInitializationError(RuntimeError):
    """Raised when shared dependencies cannot be initialized."""


def error_response(message: str, status_code: int = 500) -> JSONResponse:
    """Return a standardized JSON error response."""
    payload = RAGErrorResponse(success=False, error=message).model_dump()
    return JSONResponse(status_code=status_code, content=payload)
