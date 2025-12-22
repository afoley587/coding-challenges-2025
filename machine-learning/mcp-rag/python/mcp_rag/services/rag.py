import logging
import time
from datetime import datetime
from typing import Optional

import requests
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_postgres import PGVector
from mcp_rag.core.config import MCPServerConfig
from mcp_rag.db.pool import DatabaseConnectionPool
from psycopg2.extras import RealDictCursor


class RAGService:
    """Handles RAG operations with LLM and vector store."""

    def __init__(self, config: MCPServerConfig, db_pool: DatabaseConnectionPool):
        self.config = config
        self.db_pool = db_pool
        self.logger = logging.getLogger(f"{__name__}.RAGService")
        self._embeddings = None
        self._llm = None
        self._vectorstore = None
        self._qa_chain = None

    @property
    def embeddings(self):
        """Lazy load embeddings model."""
        if self._embeddings is None:
            self.logger.info(
                f"Initializing Ollama embeddings: {self.config.ollama_embedding_model}"
            )
            try:
                self._embeddings = OllamaEmbeddings(
                    model=self.config.ollama_embedding_model,
                    base_url=self.config.ollama_base_url,
                )
                self.logger.info("Embeddings model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load embeddings: {e}", exc_info=True)
                raise
        return self._embeddings

    @property
    def llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            self.logger.info(f"Initializing Ollama LLM: {self.config.ollama_model}")
            try:
                self._llm = OllamaLLM(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url,
                    temperature=self.config.llm_temperature,
                    timeout=self.config.ollama_timeout,
                )
                self.logger.info("LLM loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load LLM: {e}", exc_info=True)
                raise
        return self._llm

    @property
    def vectorstore(self):
        """Lazy load vectorstore."""
        if self._vectorstore is None:
            self.logger.info("Initializing PGVector vectorstore")
            self.logger.info(self.config.connection_string)
            try:
                self._vectorstore = PGVector(
                    connection=self.config.connection_string,
                    embeddings=self.embeddings,
                    collection_name="k8s_documents",
                    use_jsonb=True,
                )
                self.logger.info("Vectorstore initialized successfully")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize vectorstore: {e}", exc_info=True
                )
                raise
        return self._vectorstore

    @property
    def qa_chain(self):
        """Lazy load QA chain."""
        if self._qa_chain is None:
            self.logger.info("Initializing RAG QA chain")
            try:
                prompt_template = """You are a Kubernetes expert assistant. Use the following pieces of context from the Kubernetes documentation to answer the question.

If you don't know the answer based on the provided context, say so. Don't make up answers.

Always cite the source document and page number when providing information.

Context:
{context}

Question: {question}

Helpful Answer:"""

                prompt = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )

                self._qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": self.config.top_k_results}
                    ),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt},
                )

                self.logger.info("QA chain initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize QA chain: {e}", exc_info=True)
                raise

        return self._qa_chain

    def search_documents(self, query: str, max_results: Optional[int] = None) -> dict:
        """Search documents using RAG."""
        start_time = time.time()
        self.logger.info(f"Processing RAG query: {query[:100]}...")

        try:
            result = self.qa_chain.invoke({"query": query})

            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    metadata = doc.metadata
                    sources.append(
                        {
                            "source_file": metadata.get("source_file", "Unknown"),
                            "page_number": metadata.get("page_number", "N/A"),
                            "chunk_index": metadata.get("chunk_index", 0),
                            "excerpt": doc.page_content[:200] + "...",
                        }
                    )

            duration = time.time() - start_time
            self.logger.info(f"Query processed successfully in {duration:.2f}s")

            return {
                "success": True,
                "answer": result["result"],
                "sources": sources,
                "query": query,
                "num_sources": len(sources),
                "processing_time": duration,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }

    def semantic_search(
        self, query: str, source_file: Optional[str] = None, max_results: int = 3
    ) -> dict:
        """Perform semantic search on documents."""
        start_time = time.time()
        self.logger.info(
            f"Semantic search: {query[:100]}..."
            + (f" in {source_file}" if source_file else "")
        )

        try:
            query_embedding = self.embeddings.embed_query(query)

            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                if source_file:
                    sql = """
                        SELECT content, page_number, source_file, chunk_index,
                               1 - (embedding <=> %s::vector) as similarity
                        FROM k8s_documents
                        WHERE source_file = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """
                    cursor.execute(
                        sql,
                        (query_embedding, source_file, query_embedding, max_results),
                    )
                else:
                    sql = """
                        SELECT content, page_number, source_file, chunk_index,
                               1 - (embedding <=> %s::vector) as similarity
                        FROM k8s_documents
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """
                    cursor.execute(sql, (query_embedding, query_embedding, max_results))

                results = cursor.fetchall()
                cursor.close()

            formatted_results = [
                {
                    "text": row["content"],
                    "page_number": row["page_number"],
                    "source_file": row["source_file"],
                    "chunk_index": row["chunk_index"],
                    "relevance_score": float(row["similarity"]),
                }
                for row in results
            ]

            duration = time.time() - start_time
            self.logger.info(f"Semantic search completed in {duration:.2f}s")

            return {
                "success": True,
                "query": query,
                "source_file": source_file,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "processing_time": duration,
            }

        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}", exc_info=True)
            return {"success": False, "error": str(e), "query": query}

    def stats(self):
        start_time = time.time()

        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute("SELECT COUNT(*) as count FROM k8s_documents;")
                total_chunks = cursor.fetchone()["count"]

                cursor.execute(
                    "SELECT COUNT(DISTINCT source_file) as count FROM k8s_documents;"
                )
                unique_docs = cursor.fetchone()["count"]

                cursor.execute(
                    "SELECT DISTINCT source_file FROM k8s_documents ORDER BY source_file;"
                )
                source_files = [row["source_file"] for row in cursor.fetchall()]

                cursor.execute(
                    "SELECT pg_size_pretty(pg_database_size(current_database())) as db_size;"
                )
                db_size = cursor.fetchone()["db_size"]

                cursor.close()

            duration = time.time() - start_time

            return {
                "success": True,
                "total_chunks": total_chunks,
                "unique_documents": unique_docs,
                "source_files": source_files,
                "database_size": db_size,
                "embedding_model": self.config.ollama_embedding_model,
                "llm_model": self.config.ollama_model,
                "processing_time": duration,
            }

        except Exception as e:
            self.logger.error(f"Error getting stats: {e}", exc_info=True)
            raise

    def list_available_documents(self):
        start_time = time.time()

        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(
                    """
                    SELECT source_file, COUNT(*) as chunk_count,
                        COUNT(DISTINCT page_number) as page_count,
                        MIN(ingested_at) as first_ingested,
                        MAX(ingested_at) as last_updated
                    FROM k8s_documents
                    GROUP BY source_file
                    ORDER BY source_file;
                """
                )

                results = cursor.fetchall()
                cursor.close()

            documents = [
                {
                    "source_file": row["source_file"],
                    "chunk_count": row["chunk_count"],
                    "page_count": row["page_count"],
                    "first_ingested": (
                        row["first_ingested"].isoformat()
                        if row["first_ingested"]
                        else None
                    ),
                    "last_updated": (
                        row["last_updated"].isoformat() if row["last_updated"] else None
                    ),
                }
                for row in results
            ]

            duration = time.time() - start_time

            return {
                "success": True,
                "total_documents": len(documents),
                "documents": documents,
                "processing_time": duration,
            }
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}", exc_info=True)
            raise

    def get_health_status(self) -> dict:
        """
        Check health of all service dependencies.

        Returns:
            Dictionary with health status of all components
        """
        health = {
            "status": "healthy",
            "service": "k8s-docs-rag",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "postgres": False,
                "ollama": False,
                "database_accessible": False,
            },
        }

        # Check PostgreSQL connection
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                cursor.close()
            health["checks"]["postgres"] = True
            self.logger.debug("PostgreSQL health check: OK")
        except Exception as e:
            self.logger.warning(f"PostgreSQL health check failed: {e}")
            health["status"] = "unhealthy"

        # Check Ollama API
        try:
            response = requests.get(
                f"{self.config.ollama_base_url}/api/tags", timeout=5
            )
            health["checks"]["ollama"] = response.status_code == 200
            self.logger.debug(
                f"Ollama health check: {'OK' if health['checks']['ollama'] else 'FAILED'}"
            )
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            health["checks"]["ollama"] = False

        # Check database has documents
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM k8s_documents LIMIT 1;")
                count = cursor.fetchone()[0]
                health["checks"]["database_accessible"] = True
                health["document_count"] = count
                cursor.close()
            self.logger.debug("Database accessibility check: OK")
        except Exception as e:
            self.logger.warning(f"Database accessibility check failed: {e}")

        # Overall health
        health["healthy"] = all(health["checks"].values())

        return health
