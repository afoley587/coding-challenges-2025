#!/usr/bin/env python3
"""
Kubernetes Documentation Ingestion Script - PostgreSQL + pgvector
Production-grade PDF ingestion pipeline with configuration management and proper logging
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import psycopg2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp_rag.core.config import IngestionConfig
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class PostgresVectorStore:
    """Handles PostgreSQL connection pooling and vector operations."""

    def __init__(self, config: IngestionConfig):
        """
        Initialize connection pool to PostgreSQL.

        Args:
            config: Ingestion configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PostgresVectorStore")
        self.connection_pool = None
        self._initialize_pool()
        self._initialize_schema()

    def _initialize_pool(self):
        """Initialize connection pool with retry logic."""
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                self.logger.info(
                    f"Connecting to PostgreSQL at {self.config.postgres_host}:{self.config.postgres_port} "
                    f"(attempt {attempt}/{self.config.retry_attempts})"
                )

                self.connection_pool = pool.SimpleConnectionPool(
                    minconn=self.config.postgres_pool_min,
                    maxconn=self.config.postgres_pool_max,
                    host=self.config.postgres_host,
                    port=self.config.postgres_port,
                    database=self.config.postgres_db,
                    user=self.config.postgres_user,
                    password=self.config.postgres_password,
                )

                self.logger.info("Successfully connected to PostgreSQL")
                return

            except psycopg2.OperationalError as e:
                self.logger.warning(
                    f"Failed to connect (attempt {attempt}/{self.config.retry_attempts}): {e}"
                )
                if attempt < self.config.retry_attempts:
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error("Max connection attempts reached, aborting")
                    raise

    @contextmanager
    def _connection(self, *, cursor_factory=None):
        """Context manager that yields a pooled connection and cursor."""

        if not self.connection_pool:
            raise RuntimeError("Connection pool not initialized")

        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield conn, cursor
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def _initialize_schema(self):
        """Create tables and install pgvector extension."""
        try:
            with self._connection() as (conn, cursor):
                self.logger.info("Installing pgvector extension...")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                self.logger.info("Creating k8s_documents table...")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS k8s_documents (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(%s),
                        source_file TEXT NOT NULL,
                        file_path TEXT,
                        page_number INTEGER,
                        chunk_index INTEGER,
                        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB,
                        CONSTRAINT valid_page_number CHECK (page_number > 0),
                        CONSTRAINT valid_chunk_index CHECK (chunk_index >= 0)
                    );
                """
                    % self.config.embedding_dimension
                )

                self.logger.info("Creating indexes...")
                # Vector similarity index
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS k8s_documents_embedding_idx
                    ON k8s_documents
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """
                )

                # Source file index for filtering
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS k8s_documents_source_idx
                    ON k8s_documents (source_file);
                """
                )

                # Ingestion timestamp index for tracking
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS k8s_documents_ingested_idx
                    ON k8s_documents (ingested_at DESC);
                """
                )

                conn.commit()
                self.logger.info("Database schema initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize schema: {e}")
            raise

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Insert document chunks with embeddings.

        Args:
            chunks: List of chunk dictionaries with embeddings

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if not chunks:
            self.logger.warning("No chunks to insert")
            return 0, 0

        inserted = 0
        updated = 0

        try:
            with self._connection() as (conn, cursor):
                values = self._build_chunk_values(chunks)

                execute_values(
                    cursor,
                    """
                    INSERT INTO k8s_documents
                    (id, content, embedding, source_file, file_path, page_number, chunk_index, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        ingested_at = CURRENT_TIMESTAMP
                    RETURNING (xmax = 0) AS inserted
                    """,
                    values,
                )

                # Count inserts vs updates
                results = cursor.fetchall()
                inserted = sum(1 for r in results if r[0])
                updated = len(results) - inserted

                conn.commit()
                self.logger.debug(
                    f"Batch insert: {inserted} new, {updated} updated, {len(chunks)} total"
                )

        except Exception as e:
            self.logger.error(f"Failed to insert chunks: {e}")
            raise

        return inserted, updated

    def _build_chunk_values(self, chunks: Iterable[dict[str, Any]]):
        """Prepare chunk values for insertion while validating required fields."""

        values = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            try:
                values.append(
                    (
                        chunk["id"],
                        chunk["text"],
                        chunk["embedding"],
                        metadata["source_file"],
                        metadata.get("file_path"),
                        metadata.get("page_number"),
                        metadata.get("chunk_index"),
                        metadata.get("metadata_json"),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing required chunk field: {exc.args[0]}"
                ) from exc

        return values

    def get_stats(self) -> dict:
        """Get database statistics."""
        try:
            with self._connection(cursor_factory=RealDictCursor) as (conn, cursor):
                # Total chunks
                cursor.execute("SELECT COUNT(*) as count FROM k8s_documents;")
                total_chunks = cursor.fetchone()["count"]

                # Unique documents
                cursor.execute(
                    "SELECT COUNT(DISTINCT source_file) as count FROM k8s_documents;"
                )
                unique_docs = cursor.fetchone()["count"]

                # List of source files with counts
                cursor.execute(
                    """
                    SELECT
                        source_file,
                        COUNT(*) as chunk_count,
                        MIN(ingested_at) as first_ingested,
                        MAX(ingested_at) as last_updated
                    FROM k8s_documents
                    GROUP BY source_file
                    ORDER BY source_file;
                """
                )
                source_files = cursor.fetchall()

                # Database size
                cursor.execute(
                    """
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size;
                """
                )
                db_size = cursor.fetchone()["db_size"]

                return {
                    "total_chunks": total_chunks,
                    "unique_documents": unique_docs,
                    "source_files": [dict(f) for f in source_files],
                    "database_size": db_size,
                }

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            raise

    def reset(self):
        """Delete all documents."""
        try:
            self.logger.warning("Resetting database - all data will be deleted")
            with self._connection() as (conn, cursor):
                cursor.execute("TRUNCATE TABLE k8s_documents;")
                conn.commit()
                self.logger.info("Database reset complete")
        except Exception as e:
            self.logger.error(f"Failed to reset database: {e}")
            raise

    def close(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Connection pool closed")


class K8sDocumentIngestor:
    """Handles ingestion of Kubernetes documentation PDFs."""

    def __init__(self, config: IngestionConfig):
        """
        Initialize the ingestor with configuration.

        Args:
            config: Ingestion configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.K8sDocumentIngestor")

        self.logger.info("Initializing K8s Document Ingestor")
        self.logger.debug(f"Configuration: {config.to_dict()}")

        # Initialize database
        self.db = PostgresVectorStore(config)

        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {config.embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(config.embedding_model)
            self.logger.info(
                f"Embedding model loaded (dimension: {config.embedding_dimension})"
            )
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.logger.debug(
            f"Text splitter initialized (size: {config.chunk_size}, "
            f"overlap: {config.chunk_overlap})"
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> list[dict[str, any]]:
        """
        Extract text from PDF with page numbers and metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page data dictionaries
        """
        pages_data = []

        self._validate_pdf_path(pdf_path)

        try:
            self.logger.info(f"Opening PDF: {pdf_path.name}")
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)

                self.logger.debug(f"PDF contains {total_pages} pages")

                for page_num in range(total_pages):
                    page = doc[page_num]
                    text = page.get_text()

                    if not text.strip():
                        self.logger.debug(f"Skipping empty page {page_num + 1}")
                        continue

                    pages_data.append(
                        {
                            "text": text,
                            "page_number": page_num + 1,
                            "source_file": pdf_path.name,
                            "file_path": str(pdf_path.absolute()),
                        }
                    )

            non_empty = len(pages_data)
            self.logger.info(
                f"Extracted {non_empty}/{total_pages} non-empty pages from {pdf_path.name}"
            )

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}", exc_info=True)
            raise

        return pages_data

    def _validate_pdf_path(self, pdf_path: Path) -> None:
        """Raise helpful errors when the supplied path is invalid."""

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not pdf_path.is_file():
            raise ValueError(f"PDF path is not a file: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type (expected .pdf): {pdf_path}")

    def chunk_documents(self, pages_data: list[dict]) -> list[dict]:
        """
        Split document pages into chunks.

        Args:
            pages_data: List of page data from PDF extraction

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        self.logger.debug(f"Chunking {len(pages_data)} pages")

        for page_data in pages_data:
            page_chunks = self.text_splitter.split_text(page_data["text"])

            for chunk_idx, chunk_text in enumerate(page_chunks):
                # Create deterministic ID based on content
                chunk_id = hashlib.md5(
                    f"{page_data['source_file']}-{page_data['page_number']}-{chunk_idx}".encode()
                ).hexdigest()

                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "source_file": page_data["source_file"],
                            "file_path": page_data["file_path"],
                            "page_number": page_data["page_number"],
                            "chunk_index": chunk_idx,
                            "metadata_json": None,
                        },
                    }
                )

        self.logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks

        texts = [chunk["text"] for chunk in chunks]
        self.logger.info(f"Generating embeddings for {len(texts)} chunks...")

        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.config.embedding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
            )

            if embeddings is None or len(embeddings) != len(texts):
                raise RuntimeError(
                    "Embedding model returned an unexpected number of vectors"
                )

            expected_dim = self.config.embedding_dimension
            actual_dim = len(embeddings[0]) if len(embeddings) else 0
            if expected_dim and actual_dim and expected_dim != actual_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
                )

            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i].tolist()

            self.logger.info(f"Generated {len(embeddings)} embeddings")

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            raise

        return chunks

    def ingest_pdf(self, pdf_path: Path) -> dict[str, int]:
        """
        Complete ingestion pipeline for a single PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with ingestion statistics
        """
        start_time = time.time()
        self.logger.info(f"Starting ingestion pipeline for: {pdf_path.name}")

        try:
            # Extract text
            pages_data = self.extract_text_from_pdf(pdf_path)
            if not pages_data:
                self.logger.warning(f"No text extracted from {pdf_path.name}")
                return {"chunks": 0, "inserted": 0, "updated": 0, "duration": 0}

            # Chunk documents
            chunks = self.chunk_documents(pages_data)
            if not chunks:
                self.logger.warning(f"No chunks created from {pdf_path.name}")
                return {"chunks": 0, "inserted": 0, "updated": 0, "duration": 0}

            # Generate embeddings
            chunks = self.embed_chunks(chunks)

            # Insert into database in batches
            total_inserted = 0
            total_updated = 0
            batch_size = self.config.batch_insert_size

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                inserted, updated = self.db.insert_chunks(batch)
                total_inserted += inserted
                total_updated += updated

            duration = time.time() - start_time

            self.logger.info(
                f"✓ Ingestion complete for {pdf_path.name}: "
                f"{total_inserted} new, {total_updated} updated, "
                f"{len(chunks)} total chunks in {duration:.2f}s"
            )

            return {
                "chunks": len(chunks),
                "inserted": total_inserted,
                "updated": total_updated,
                "duration": duration,
            }

        except Exception as e:
            self.logger.error(f"Failed to ingest {pdf_path.name}: {e}", exc_info=True)
            raise

    def ingest_directory(self, directory: Path) -> dict[str, any]:
        """
        Ingest all PDFs in a directory.

        Args:
            directory: Path to directory containing PDFs

        Returns:
            Dictionary with ingestion statistics
        """
        pdf_files = sorted(list(directory.glob("*.pdf")))

        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory}")
            return {
                "total_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "total_inserted": 0,
                "total_updated": 0,
                "duration": 0,
                "failed_file_list": [],
            }

        self.logger.info(f"Found {len(pdf_files)} PDF files to process")

        start_time = time.time()
        total_chunks = 0
        total_inserted = 0
        total_updated = 0
        successful_files = 0
        failed_files = []
        failed_file_errors: list[dict[str, str]] = []

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            try:
                result = self.ingest_pdf(pdf_path)
                total_chunks += result["chunks"]
                total_inserted += result["inserted"]
                total_updated += result["updated"]
                successful_files += 1
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_path.name}: {e}")
                failed_files.append(pdf_path.name)
                failed_file_errors.append({"file": pdf_path.name, "error": str(e)})

        total_duration = time.time() - start_time

        return {
            "total_files": len(pdf_files),
            "successful_files": successful_files,
            "failed_files": len(failed_files),
            "total_chunks": total_chunks,
            "total_inserted": total_inserted,
            "total_updated": total_updated,
            "duration": total_duration,
            "failed_file_list": failed_files,
            "failed_file_errors": failed_file_errors,
        }

    def get_stats(self) -> dict:
        """Get database statistics."""
        stats = self.db.get_stats()
        if "error" in stats:
            raise RuntimeError(f"Failed to fetch database stats: {stats['error']}")
        stats["embedding_model"] = self.config.embedding_model
        stats["chunk_size"] = self.config.chunk_size
        return stats

    def reset(self):
        """Reset the database."""
        self.db.reset()

    def close(self):
        """Close connections."""
        self.db.close()


def setup_logging(config: IngestionConfig):
    """
    Configure logging for the application.

    Args:
        config: Ingestion configuration
    """
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=config.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            (
                logging.FileHandler("/app/logs/ingestion.log")
                if Path("/app/logs").exists()
                else logging.NullHandler()
            ),
        ],
    )

    # Set third-party loggers to WARNING
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest K8s documentation PDFs into PostgreSQL+pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all PDFs in default directory
  python ingest_k8s_docs.py

  # Ingest from specific directory
  python ingest_k8s_docs.py --docs-dir /path/to/pdfs

  # Reset database before ingesting
  python ingest_k8s_docs.py --reset

  # Show database statistics
  python ingest_k8s_docs.py --stats

  # Enable debug logging
  python ingest_k8s_docs.py --log-level DEBUG
        """,
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        help="Directory containing PDFs (default: from env DOCS_DIRECTORY)",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset the database before ingesting"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show database statistics and exit"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: from env LOG_LEVEL or INFO)",
    )

    args = parser.parse_args()

    # Override config with CLI arguments
    config_overrides = {}
    if args.docs_dir:
        config_overrides["docs_directory"] = args.docs_dir
    if args.log_level:
        config_overrides["log_level"] = args.log_level

    # Initialize configuration
    try:
        config = IngestionConfig(**config_overrides)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("K8s Documentation Ingestion Pipeline")
    logger.info("=" * 80)

    # Initialize ingestor
    try:
        ingestor = K8sDocumentIngestor(config)
    except Exception as e:
        logger.error(f"Failed to initialize ingestor: {e}", exc_info=True)
        sys.exit(1)

    try:
        # Show stats if requested
        if args.stats:
            logger.info("Retrieving database statistics...")
            stats = ingestor.get_stats()

            logger.info("=" * 80)
            logger.info("DATABASE STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Total chunks: {stats['total_chunks']}")
            logger.info(f"Unique documents: {stats['unique_documents']}")
            logger.info(f"Database size: {stats.get('database_size', 'N/A')}")
            logger.info(f"Embedding model: {stats['embedding_model']}")
            logger.info(f"Chunk size: {stats['chunk_size']}")

            if stats.get("source_files"):
                logger.info("\nSource files:")
                for file_info in stats["source_files"]:
                    logger.info(
                        f"  - {file_info['source_file']}: "
                        f"{file_info['chunk_count']} chunks, "
                        f"last updated: {file_info['last_updated']}"
                    )

            return

        # Reset if requested
        if args.reset:
            logger.warning("Reset flag detected")
            ingestor.reset()

        # Ingest documents
        docs_dir = Path(config.docs_directory)

        logger.info("=" * 80)
        logger.info("STARTING INGESTION")
        logger.info("=" * 80)
        logger.info(f"Source directory: {docs_dir}")
        logger.info(
            f"Database: {config.postgres_host}:{config.postgres_port}/{config.postgres_db}"
        )
        logger.info(f"Chunk size: {config.chunk_size}")
        logger.info(f"Chunk overlap: {config.chunk_overlap}")
        logger.info(f"Batch size: {config.batch_insert_size}")
        logger.info("=" * 80)

        results = ingestor.ingest_directory(docs_dir)

        # Print summary
        logger.info("=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(
            f"Files processed: {results['successful_files']}/{results['total_files']}"
        )
        logger.info(f"Failed files: {results['failed_files']}")
        logger.info(f"Total chunks: {results['total_chunks']}")
        logger.info(f"New chunks: {results['total_inserted']}")
        logger.info(f"Updated chunks: {results['total_updated']}")
        logger.info(f"Duration: {results['duration']:.2f}s")

        if results["failed_file_list"]:
            logger.warning(f"Failed files: {', '.join(results['failed_file_list'])}")
            for failure in results.get("failed_file_errors", []):
                logger.warning(
                    "  • %s: %s",
                    failure.get("file"),
                    failure.get("error", "unknown error"),
                )

        # Show final stats
        stats = ingestor.get_stats()
        logger.info("=" * 80)
        logger.info("FINAL DATABASE STATE")
        logger.info("=" * 80)
        logger.info(f"Total chunks in DB: {stats['total_chunks']}")
        logger.info(f"Unique documents: {stats['unique_documents']}")
        logger.info(f"Database size: {stats.get('database_size', 'N/A')}")
        logger.info("=" * 80)
        logger.info("Ready to start the MCP server!")

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error during ingestion: {e}", exc_info=True)
        sys.exit(1)
    finally:
        ingestor.close()


if __name__ == "__main__":
    main()
