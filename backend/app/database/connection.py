"""
backend/app/database/connection.py
----------------------------------------------------
Production-Grade Thread-Safe MySQL Connection Pool Manager for Genkit AI V6.
Direct, exclusive MySQL connectivity with automatic pool reconnection and graceful fallback.
"""

from typing import Any, List, Optional, Tuple
import mysql.connector
from mysql.connector import pooling, errors

from app.core.config import settings
from app.core.logger import logger
from app.database.models import MYSQL_SCHEMA


class DatabaseManager:
    """Thread-safe dedicated MySQL Database Connection Pool Manager."""

    def __init__(self):
        self.engine_type = "mysql"
        self._pool: Optional[pooling.MySQLConnectionPool] = None
        self._available: bool = False
        self._init_mysql_pool()

    def _init_mysql_pool(self) -> None:
        """Initializes the MySQL Connection Pool and applies table schema."""
        try:
            self._pool = pooling.MySQLConnectionPool(
                pool_name="genkit_mysql_pool",
                pool_size=settings.MYSQL_MIN_POOL_SIZE,
                pool_reset_session=True,
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=settings.MYSQL_DATABASE,
            )

            # Validate connection & apply tables
            conn = self._pool.get_connection()
            try:
                cursor = conn.cursor()
                for statement in MYSQL_SCHEMA.strip().split(";"):
                    stmt = statement.strip()
                    if stmt:
                        cursor.execute(stmt)
                conn.commit()
                cursor.close()
                self._available = True
                logger.info(
                    f"Successfully connected to MySQL database '{settings.MYSQL_DATABASE}' "
                    f"at {settings.MYSQL_HOST}:{settings.MYSQL_PORT}"
                )
            finally:
                conn.close()

        except Exception as e:
            self._pool = None
            self._available = False
            logger.warning(
                f"MySQL database is currently unavailable at {settings.MYSQL_HOST}:{settings.MYSQL_PORT} "
                f"({e}). Running in offline/in-memory degradation mode."
            )

    @property
    def is_available(self) -> bool:
        """Returns True if MySQL connection pool is healthy."""
        return self._available and self._pool is not None

    def get_connection(self):
        """Retrieves a healthy connection from the MySQL pool."""
        if not self._pool:
            self._init_mysql_pool()
        if not self._pool:
            raise ConnectionError("MySQL database is offline.")
        return self._pool.get_connection()

    def execute_write(self, query: str, params: Tuple[Any, ...] = ()) -> Optional[int]:
        """Executes an INSERT / UPDATE / DELETE write query with parameterized inputs."""
        try:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                last_id = cursor.lastrowid
                cursor.close()
                return last_id
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"MySQL write operation skipped (database offline or failed): {e}")
            return None

    def execute_read(self, query: str, params: Tuple[Any, ...] = ()) -> List[dict]:
        """Executes a SELECT read query with parameterized inputs and returns dict rows."""
        try:
            conn = self.get_connection()
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                cursor.close()
                return rows
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"MySQL read operation skipped (database offline or failed): {e}")
            return []


# Instantiated Singleton Database Manager
db_manager = DatabaseManager()
