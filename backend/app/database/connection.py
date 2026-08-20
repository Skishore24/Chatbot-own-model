"""
backend/app/database/connection.py
----------------------------------------------------
Production-Grade Thread-Safe MySQL Connection Pool Manager for Genkit AI V6.
Direct, exclusive MySQL connectivity with automatic pool reconnection.
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
                logger.info(
                    f"Successfully connected to MySQL database '{settings.MYSQL_DATABASE}' "
                    f"at {settings.MYSQL_HOST}:{settings.MYSQL_PORT}"
                )
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize MySQL Connection Pool: {e}")
            raise ConnectionError(
                f"MySQL connection failed to {settings.MYSQL_HOST}:{settings.MYSQL_PORT} "
                f"database '{settings.MYSQL_DATABASE}'. Error: {e}"
            )

    def get_connection(self):
        """Retrieves a healthy connection from the MySQL pool."""
        if not self._pool:
            self._init_mysql_pool()
        return self._pool.get_connection()

    def execute_write(self, query: str, params: Tuple[Any, ...] = ()) -> Optional[int]:
        """Executes an INSERT / UPDATE / DELETE write query with parameterized inputs."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            last_id = cursor.lastrowid
            cursor.close()
            return last_id
        except Exception as e:
            logger.error(f"MySQL write operation failed: {e} | Query: {query} | Params: {params}")
            raise e
        finally:
            conn.close()

    def execute_read(self, query: str, params: Tuple[Any, ...] = ()) -> List[dict]:
        """Executes a SELECT read query with parameterized inputs and returns dict rows."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Exception as e:
            logger.error(f"MySQL read operation failed: {e} | Query: {query} | Params: {params}")
            raise e
        finally:
            conn.close()


# Instantiated Singleton Database Manager
db_manager = DatabaseManager()
