import logging
import time
from contextlib import contextmanager

import psycopg2
from mcp_rag.core.config import MCPServerConfig
from psycopg2 import pool


class DatabaseConnectionPool:
    """Manages PostgreSQL connection pool."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DatabaseConnectionPool")
        self.pool = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool with retry logic."""
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                self.logger.info(
                    f"Connecting to PostgreSQL at {self.config.postgres_host}:{self.config.postgres_port} "
                    f"(attempt {attempt}/{self.config.retry_attempts})"
                )

                self.pool = pool.ThreadedConnectionPool(
                    minconn=self.config.postgres_pool_min,
                    maxconn=self.config.postgres_pool_max,
                    **self.config.connection_params,
                )

                conn = self.pool.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                cursor.close()
                self.pool.putconn(conn)

                self.logger.info("Successfully connected to PostgreSQL")
                return

            except psycopg2.Error as e:
                self.logger.warning(
                    f"Failed to connect (attempt {attempt}/{self.config.retry_attempts}): {e}"
                )
                if attempt < self.config.retry_attempts:
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error("Max connection attempts reached")
                    raise

    @contextmanager
    def get_connection(self):
        """Context manager for getting a database connection."""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database operation failed: {e}", exc_info=True)
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            self.logger.info("Connection pool closed")
