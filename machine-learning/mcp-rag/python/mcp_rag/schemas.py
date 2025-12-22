from typing import Optional

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    service: str = Field("k8s-docs-rag", description="Application name")
    version: str = Field("1.0.0", description="API Version")


class HealthCheck(BaseModel):
    """Health check response model."""

    postgres: bool = Field(description="PostgreSQL connection status")
    ollama: bool = Field(description="Ollama API availability")
    database_accessible: bool = Field(description="Database query accessibility")


class HealthResponse(BaseModel):
    """Health endpoint response model."""

    status: str = Field(description="Overall status: 'healthy' or 'unhealthy'")
    service: str = Field(description="Service name")
    timestamp: str = Field(description="ISO 8601 timestamp")
    checks: HealthCheck = Field(description="Individual component health checks")
    healthy: bool = Field(description="Overall health status")
    document_count: Optional[int] = Field(
        None, description="Number of documents in database"
    )


class ReadinessCheck(BaseModel):
    """Readiness check response model."""

    embeddings_loaded: bool = Field(description="Embeddings model loaded status")
    llm_loaded: bool = Field(description="LLM loaded status")
    vectorstore_ready: bool = Field(description="Vectorstore ready status")


class ReadinessResponse(BaseModel):
    """Readiness endpoint response model."""

    ready: bool = Field(description="Overall readiness status")
    timestamp: str = Field(description="ISO 8601 timestamp")
    checks: ReadinessCheck = Field(description="Individual component readiness checks")


class ErrorResponse(BaseModel):
    """Error response model."""

    status: str = Field(default="unhealthy", description="Error status")
    error: str = Field(description="Error message")
    ready: Optional[bool] = Field(
        None, description="Readiness status for readiness endpoint"
    )


class DocumentSource(BaseModel):
    """Source document reference."""

    source_file: str = Field(description="Name of the source PDF file")
    page_number: int | str = Field(description="Page number in the document")
    chunk_index: int = Field(description="Index of the chunk within the page")
    excerpt: str = Field(description="Brief excerpt from the chunk")


class SearchResponse(BaseModel):
    """RAG search response model."""

    success: bool = Field(description="Whether the search was successful")
    answer: str = Field(description="Generated answer from the LLM")
    sources: list[DocumentSource] = Field(
        description="Source documents used for the answer"
    )
    query: str = Field(description="Original search query")
    num_sources: int = Field(description="Number of source documents")
    processing_time: float = Field(
        description="Time taken to process the query in seconds"
    )
    timestamp: str = Field(description="ISO 8601 timestamp")


class RAGErrorResponse(BaseModel):
    """Error response for RAG operations."""

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    timestamp: str = Field(description="ISO 8601 timestamp")


class SemanticSearchResult(BaseModel):
    """Individual semantic search result."""

    text: str = Field(description="Content of the matched chunk")
    page_number: int | str = Field(description="Page number")
    source_file: str = Field(description="Source file name")
    chunk_index: int = Field(description="Chunk index")
    relevance_score: float = Field(description="Relevance score (0-1)")


class SemanticSearchResponse(BaseModel):
    """Semantic search response model."""

    success: bool = Field(description="Whether the search was successful")
    query: str = Field(description="Search query")
    source_file: Optional[str] = Field(
        None, description="Source file filter if applied"
    )
    results: list[SemanticSearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results")
    processing_time: float = Field(description="Processing time in seconds")


class DocumentInfo(BaseModel):
    """Document metadata."""

    source_file: str = Field(description="Name of the source file")
    chunk_count: int = Field(description="Number of chunks from this document")
    page_count: int = Field(description="Number of pages")
    first_ingested: Optional[str] = Field(None, description="First ingestion timestamp")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class DocumentListResponse(BaseModel):
    """List of available documents response."""

    success: bool = Field(description="Whether the operation was successful")
    total_documents: int = Field(description="Total number of documents")
    documents: list[DocumentInfo] = Field(description="List of document metadata")
    processing_time: float = Field(description="Processing time in seconds")


class DatabaseStatsResponse(BaseModel):
    """Database statistics response."""

    success: bool = Field(description="Whether the operation was successful")
    total_chunks: int = Field(description="Total number of chunks in database")
    unique_documents: int = Field(description="Number of unique documents")
    source_files: list[str] = Field(description="List of source file names")
    database_size: str = Field(description="Database size (human readable)")
    embedding_model: str = Field(description="Embedding model name")
    llm_model: str = Field(description="LLM model name")
    processing_time: float = Field(description="Processing time in seconds")
