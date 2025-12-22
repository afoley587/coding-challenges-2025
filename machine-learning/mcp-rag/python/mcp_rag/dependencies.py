"""Application dependency wiring for MCP RAG services."""

from __future__ import annotations

import logging
from typing import Optional

from mcp_rag.core.config import MCPServerConfig
from mcp_rag.db.pool import DatabaseConnectionPool
from mcp_rag.services.rag import RAGService

logger = logging.getLogger(__name__)

_config: Optional[MCPServerConfig] = None
_db_pool: Optional[DatabaseConnectionPool] = None
_rag_service: Optional[RAGService] = None


def get_config() -> MCPServerConfig:
    """Return a shared MCPServerConfig instance."""
    global _config

    if _config is None:
        _config = MCPServerConfig()
        logger.info("Configuration initialized for MCP server")
    return _config


def get_db_pool() -> DatabaseConnectionPool:
    """Return a shared database connection pool."""
    global _db_pool

    if _db_pool is None:
        _db_pool = DatabaseConnectionPool(get_config())
        logger.info("Database connection pool initialized")
    return _db_pool


def get_rag_service() -> RAGService:
    """Return a shared RAGService instance."""
    global _rag_service

    if _rag_service is None:
        _rag_service = RAGService(get_config(), get_db_pool())
        logger.info("RAG service initialized")
    return _rag_service


def close_resources() -> None:
    """Close shared resources such as the connection pool."""
    global _db_pool

    if _db_pool:
        _db_pool.close()
        _db_pool = None
