import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""

    # PostgreSQL configuration
    postgres_host: str = field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "postgres-service")
    )
    postgres_port: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432"))
    )
    postgres_db: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "k8s_docs")
    )
    postgres_user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "postgres")
    )
    postgres_password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres")
    )

    # Connection pool settings
    postgres_pool_min: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_POOL_MIN", "2"))
    )
    postgres_pool_max: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_POOL_MAX", "10"))
    )

    # Document processing configuration
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )

    # Embedding configuration
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_dimension: int = 384  # Dimension for all-MiniLM-L6-v2
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )

    # Ingestion configuration
    docs_directory: str = field(
        default_factory=lambda: os.getenv("DOCS_DIRECTORY", "/app/k8s_docs")
    )
    batch_insert_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_INSERT_SIZE", "100"))
    )
    retry_attempts: int = field(
        default_factory=lambda: int(os.getenv("RETRY_ATTEMPTS", "3"))
    )
    retry_delay: int = field(default_factory=lambda: int(os.getenv("RETRY_DELAY", "5")))

    # Logging configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = (
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration values."""
        if self.chunk_size < 100:
            raise ValueError(f"chunk_size must be >= 100, got {self.chunk_size}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

        if self.embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be > 0, got {self.embedding_dimension}"
            )

        if not Path(self.docs_directory).exists():
            raise ValueError(f"docs_directory does not exist: {self.docs_directory}")

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return (
            f"host={self.postgres_host} "
            f"port={self.postgres_port} "
            f"dbname={self.postgres_db} "
            f"user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "postgres_host": self.postgres_host,
            "postgres_port": self.postgres_port,
            "postgres_db": self.postgres_db,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "docs_directory": self.docs_directory,
            "batch_insert_size": self.batch_insert_size,
        }


@dataclass
class MCPServerConfig:
    """Configuration for the MCP server."""

    # PostgreSQL configuration
    postgres_host: str = field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "postgres-service")
    )
    postgres_port: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432"))
    )
    postgres_db: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "k8s_docs")
    )
    postgres_user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "postgres")
    )
    postgres_password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres")
    )

    # Connection pool settings
    postgres_pool_min: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_POOL_MIN", "2"))
    )
    postgres_pool_max: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_POOL_MAX", "10"))
    )

    # Ollama configuration
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", "http://ollama-service:11434"
        )
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    )
    ollama_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"
        )
    )
    ollama_timeout: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "120"))
    )

    # RAG configuration
    top_k_results: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "5"))
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    max_context_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_LENGTH", "4096"))
    )

    # Server configuration
    mcp_server_name: str = field(
        default_factory=lambda: os.getenv("MCP_SERVER_NAME", "k8s-docs-rag")
    )
    enable_health_check: bool = field(
        default_factory=lambda: os.getenv("ENABLE_HEALTH_CHECK", "true").lower()
        == "true"
    )

    # Logging configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = (
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )

    # Retry configuration
    retry_attempts: int = field(
        default_factory=lambda: int(os.getenv("RETRY_ATTEMPTS", "3"))
    )
    retry_delay: int = field(default_factory=lambda: int(os.getenv("RETRY_DELAY", "5")))

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration values."""
        if self.top_k_results < 1:
            raise ValueError(f"top_k_results must be >= 1, got {self.top_k_results}")

        if not 0 <= self.llm_temperature <= 2:
            raise ValueError(
                f"llm_temperature must be between 0 and 2, got {self.llm_temperature}"
            )

        if self.postgres_pool_min < 1:
            raise ValueError(
                f"postgres_pool_min must be >= 1, got {self.postgres_pool_min}"
            )

        if self.postgres_pool_max < self.postgres_pool_min:
            raise ValueError(
                f"postgres_pool_max ({self.postgres_pool_max}) must be >= "
                f"postgres_pool_min ({self.postgres_pool_min})"
            )

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string for PGVector."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def connection_params(self) -> dict[str, Any]:
        """Get PostgreSQL connection parameters for psycopg2."""
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "database": self.postgres_db,
            "user": self.postgres_user,
            "password": self.postgres_password,
        }

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging (without sensitive data)."""
        return {
            "postgres_host": self.postgres_host,
            "postgres_port": self.postgres_port,
            "postgres_db": self.postgres_db,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "ollama_embedding_model": self.ollama_embedding_model,
            "top_k_results": self.top_k_results,
            "llm_temperature": self.llm_temperature,
            "mcp_server_name": self.mcp_server_name,
        }
