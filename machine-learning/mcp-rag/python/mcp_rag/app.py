#!/usr/bin/env python3
"""
Kubernetes Documentation RAG MCP Server - PostgreSQL + pgvector
Production-grade MCP server with FastAPI integration, health checks, CORS,
contextual logging, elicitation, resources, and prompts
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends as FastMCPDepends
from mcp_rag.core.config import MCPServerConfig
from mcp_rag.db.pool import DatabaseConnectionPool
from mcp_rag.schemas import (
    DatabaseStatsResponse,
    DocumentListResponse,
    HealthResponse,
    RAGErrorResponse,
    ReadinessResponse,
    RootResponse,
    SearchResponse,
    SemanticSearchResponse,
)
from mcp_rag.services.rag import RAGService

logger = logging.getLogger(__name__)

mcp = FastMCP("k8s-docs-rag")

fastmcp_http = mcp.http_app(stateless_http=True)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    logger.info("Starting up the app...")

    yield
    logger.info("Shutting down the app...")

    global _db_pool
    if _db_pool:
        _db_pool.close()


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    # Run both lifespans
    async with app_lifespan(app):
        async with fastmcp_http.lifespan(app):
            yield


api = FastAPI(
    title="K8s Documentation RAG API",
    description="MCP server for Kubernetes documentation with RAG",
    version="1.0.0",
    lifespan=combined_lifespan,
)

_config = None
_db_pool = None
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or initialize RAG service."""
    global _rag_service, _config, _db_pool

    if _rag_service is None:
        if _config is None:
            _config = MCPServerConfig()
        if _db_pool is None:
            _db_pool = DatabaseConnectionPool(_config)
        _rag_service = RAGService(_config, _db_pool)

    return _rag_service


@api.get("/", response_model=RootResponse)
async def root():
    """Root endpoint."""
    return RootResponse()


@api.get("/health", response_model=HealthResponse)
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    """
    Kubernetes liveness probe.
    Checks if the service is alive and responding.
    """
    try:
        health_data = rag_service.get_health_status()
        health_response = HealthResponse(**health_data)

        status_code = 200 if health_response.healthy else 503
        return JSONResponse(
            content=health_response.model_dump(), status_code=status_code
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)


@api.get("/readiness", response_model=ReadinessResponse)
async def readiness_check(rag_service: RAGService = Depends(get_rag_service)):
    """
    Kubernetes readiness probe.
    Checks if the service is ready to accept traffic.
    """
    try:
        readiness_data = {
            "ready": True,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "embeddings_loaded": rag_service.embeddings is not None,
                "llm_loaded": rag_service.llm is not None,
                "vectorstore_ready": rag_service.vectorstore is not None,
            },
        }
        readiness_data["ready"] = all(readiness_data["checks"].values())

        readiness_response = ReadinessResponse(**readiness_data)

        status_code = 200 if readiness_response.ready else 503
        return JSONResponse(readiness_response.model_dump(), status_code=status_code)

    except Exception as e:
        return JSONResponse({"ready": False, "error": str(e)}, status_code=503)


@mcp.tool()
async def search_k8s_docs(
    query: str,
    ctx: Context,
    max_results: int = 5,
    rag_service: RAGService = FastMCPDepends(get_rag_service),
) -> SearchResponse:
    """
    Search Kubernetes documentation using semantic search and RAG.

    Args:
        query: The question or search query about Kubernetes
        max_results: Maximum number of relevant chunks to retrieve
        ctx: MCP context for logging and progress

    Returns:
        Dictionary containing the answer and source documents
    """

    await ctx.info(f"Searching K8s docs for: {query[:80]}...")
    await ctx.report_progress(0, 100, "Querying vector database...")

    result = rag_service.search_documents(query, max_results)

    if result["success"]:
        await ctx.report_progress(100, 100, "Search complete")
        await ctx.debug(f"Found {result['num_sources']} source documents")

    return SearchResponse(**result)


@mcp.tool()
async def search_with_confirmation(
    query: str, ctx: Context, rag_service: RAGService = FastMCPDepends(get_rag_service)
) -> SearchResponse | RAGErrorResponse:
    """
    Search K8s docs with optional user confirmation using elicitation.

    Args:
        query: The search query
        ctx: MCP context for elicitation

    Returns:
        Search results after optional confirmation
    """

    # Use elicitation to confirm the query
    await ctx.info("Preparing to search Kubernetes documentation...")

    result = await ctx.elicit(
        message=f"I'm about to search for: '{query}'. Do you want to proceed?",
        response_type=str,
    )

    if result.action == "reject" or not result.data:
        await ctx.warning("Search cancelled by user")
        return RAGErrorResponse(
            success=False,
            error="Search cancelled by user",
            timestamp=datetime.now().isoformat(),
        )

    await ctx.info("Search confirmed, proceeding...")

    result = rag_service.search_documents(query)

    return SearchResponse(**result)


@mcp.tool()
async def get_database_stats(
    ctx: Context, rag_service: RAGService = FastMCPDepends(get_rag_service)
) -> DatabaseStatsResponse | RAGErrorResponse:
    """
    Get statistics about the Kubernetes documentation database.
    """

    await ctx.info("Retrieving database statistics...")

    try:
        stats = rag_service.stats()
        return DatabaseStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        await ctx.error(f"Failed to retrieve stats: {str(e)}")
        return RAGErrorResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


@mcp.tool()
async def list_available_documents(
    ctx: Context, rag_service: RAGService = FastMCPDepends(get_rag_service)
) -> DocumentListResponse | RAGErrorResponse:
    """List all available Kubernetes documentation files."""

    await ctx.info("Listing available K8s documentation files...")

    try:
        result = rag_service.list_available_documents()

        await ctx.debug(
            f"Found {len(result['documents'])} documents in {result['duration']:.2f}s"
        )

        return DocumentListResponse(**result)

    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        await ctx.error(f"Failed to list documents: {str(e)}")
        return RAGErrorResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


@mcp.tool()
async def search_by_document(
    ctx: Context,
    query: str,
    source_file: str,
    max_results: int = 3,
    rag_service: RAGService = FastMCPDepends(get_rag_service),
) -> SemanticSearchResponse | RAGErrorResponse:
    """
    Search within a specific Kubernetes documentation file.

    Args:
        query: The search query
        source_file: Name of the specific PDF file
        max_results: Maximum number of results
        ctx: MCP context for logging
    """

    if ctx:
        await ctx.info(f"Searching in {source_file}...")

    result = rag_service.semantic_search(query, source_file, max_results)

    if ctx and result["success"]:
        await ctx.debug(f"Found {result['total_results']} results in {source_file}")

    return SemanticSearchResponse(**result)


@mcp.resource("k8s://stats")
async def k8s_stats_resource(
    ctx: Context, rag_service: RAGService = FastMCPDepends(get_rag_service)
) -> str:
    """
    Resource providing current database statistics.
    LLM can read this to understand what's available.
    """
    await ctx.debug("Reading k8s://stats resource")
    stats = rag_service.stats()

    if not stats["success"]:
        return f"Error: {stats.get('error', 'Unknown error')}"

    return f"""Kubernetes Documentation Database Statistics:
- Total indexed chunks: {stats['total_chunks']}
- Unique documents: {stats['unique_documents']}
- Database size: {stats['database_size']}
- Embedding model: {stats['embedding_model']}
- LLM model: {stats['llm_model']}

Available documents:
{chr(10).join(f"  - {doc}" for doc in stats['source_files'])}
"""


# ============================================================================
# MCP PROMPTS - Reusable prompt templates
# ============================================================================


@mcp.prompt()
async def k8s_troubleshooting(error_message: str, ctx: Context) -> list[dict]:
    """
    Generate a troubleshooting prompt for Kubernetes errors.

    Args:
        error_message: The error message or symptom

    Returns:
        List of messages for the LLM
    """
    await ctx.debug("Generating K8s troubleshooting prompt")

    return [
        {
            "role": "system",
            "content": "You are a Kubernetes troubleshooting expert. Use the K8s documentation to help diagnose and resolve issues.",
        },
        {
            "role": "user",
            "content": f"""I'm experiencing this Kubernetes issue:

{error_message}

Please help me:
1. Understand what this error means
2. Identify likely root causes
3. Suggest troubleshooting steps
4. Provide solutions or workarounds

Use the search_k8s_docs tool to find relevant documentation.""",
        },
    ]


@mcp.prompt()
async def k8s_best_practices(resource_type: str, ctx: Context) -> list[dict]:
    """
    Generate a prompt for K8s best practices.

    Args:
        resource_type: Type of K8s resource (pod, deployment, service, etc.)
    """
    await ctx.debug(f"Generating best practices prompt for {resource_type}")

    return [
        {
            "role": "system",
            "content": "You are a Kubernetes expert focused on best practices and production readiness.",
        },
        {
            "role": "user",
            "content": f"""What are the best practices for {resource_type} in Kubernetes?

Please search the documentation and provide:
1. Production-ready configuration examples
2. Security considerations
3. Resource management recommendations
4. Common pitfalls to avoid
5. Monitoring and observability best practices

Use the search_k8s_docs tool to find authoritative guidance.""",
        },
    ]


@mcp.prompt()
async def k8s_migration_guide(
    from_version: str, to_version: str, ctx: Context
) -> list[dict]:
    """
    Generate a prompt for K8s version migration guidance.

    Args:
        from_version: Current Kubernetes version
        to_version: Target Kubernetes version
    """
    return [
        {"role": "system", "content": "You are a Kubernetes migration specialist."},
        {
            "role": "user",
            "content": f"""I need to migrate from Kubernetes {from_version} to {to_version}.

Please help me:
1. Identify breaking changes and deprecations
2. List new features I should be aware of
3. Provide migration steps and considerations
4. Highlight API version changes

Use the search_k8s_docs tool to find version-specific documentation.""",
        },
    ]


def setup_logging(config: MCPServerConfig):
    """Configure logging for the application."""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    if os.path.exists("/app/logs"):
        handlers.append(logging.FileHandler("/app/logs/mcp-server.log"))

    logging.basicConfig(level=log_level, format=config.log_format, handlers=handlers)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    global _config

    try:
        _config = MCPServerConfig()
        config = _config
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config)

    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")

    logger.info(f"Server name: {config.mcp_server_name}")
    logger.info(f"CORS origins: {cors_origins}")
    logger.info(
        f"PostgreSQL: {config.postgres_host}:{config.postgres_port}/{config.postgres_db}"
    )
    logger.info(f"Ollama: {config.ollama_base_url}")
    logger.info(f"LLM Model: {config.ollama_model}")
    logger.info(f"Embedding Model: {config.ollama_embedding_model}")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "mcp-session-id",
            "mcp-protocol-version",
        ],
        expose_headers=["mcp-session-id", "X-Request-Id"],
        max_age=3600,
    )
    api.mount("/mcp", fastmcp_http)

    logger.info("FastAPI application configured")

    logger.info("Pre-initializing services...")
    try:
        get_rag_service()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)

    return api


# Create the app instance
app = create_app()

if __name__ == "__main__":

    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8080"))
    workers = int(os.getenv("UVICORN_WORKERS", "1"))

    logger.info(f"Starting uvicorn on {host}:{port} with {workers} worker(s)")

    uvicorn.run(
        api, host=host, port=port, workers=workers, log_level="info", access_log=True
    )
