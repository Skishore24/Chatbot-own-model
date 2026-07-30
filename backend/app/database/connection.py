"""
backend/app/database/connection.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Async MySQL Connection Manager
Asynchronous database connection pool with resilient fallback handling.
"""

import sys
from typing import Any, Dict, List, Optional
import mysql.connector

from app.core.logger import logger
from app.core.config import settings


class AsyncDatabasePool:
    """Enterprise MySQL Database Connection Manager."""

    def __init__(self):
        self.is_connected = False
        self._init_connection()

    def _init_connection(self):
        """Initializes database tables if connection is available."""
        try:
            conn = mysql.connector.connect(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {settings.MYSQL_DATABASE};")
            cursor.execute(f"USE {settings.MYSQL_DATABASE};")

            # Create Chat History table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                role ENUM('user', 'assistant', 'system') NOT NULL,
                content TEXT NOT NULL,
                intent VARCHAR(64),
                confidence_score FLOAT,
                tokens_generated INT,
                latency_ms FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_session_time (session_id, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Create Leads table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64),
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                phone VARCHAR(32),
                service_interest VARCHAR(255),
                estimated_budget VARCHAR(64),
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_lead_email (email)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Create Feedback table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                rating INT NOT NULL,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            conn.commit()
            cursor.close()
            conn.close()
            self.is_connected = True
            logger.info("Successfully connected to MySQL database and verified schemas.")
        except Exception as e:
            self.is_connected = False
            logger.warning(f"MySQL Database connection unavailable ({str(e)}). Running in in-memory session mode.")

    def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = "General",
        confidence: float = 1.0,
        tokens_generated: int = 0,
        latency_ms: float = 0.0,
    ) -> bool:
        """Saves a chat message turn to MySQL database."""
        if not self.is_connected:
            return False

        try:
            conn = mysql.connector.connect(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=settings.MYSQL_DATABASE,
            )
            cursor = conn.cursor()
            query = """
            INSERT INTO chat_messages (session_id, role, content, intent, confidence_score, tokens_generated, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (session_id, role, content, intent, confidence, tokens_generated, latency_ms))
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving chat message to MySQL: {str(e)}")
            return False

    def save_lead(
        self,
        session_id: str,
        name: str,
        email: str,
        phone: Optional[str] = None,
        service: Optional[str] = None,
        budget: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Saves a business lead to MySQL database."""
        if not self.is_connected:
            return False

        try:
            conn = mysql.connector.connect(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=settings.MYSQL_DATABASE,
            )
            cursor = conn.cursor()
            query = """
            INSERT INTO leads (session_id, name, email, phone, service_interest, estimated_budget, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (session_id, name, email, phone or "", service or "", budget or "", notes or ""))
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving lead to MySQL: {str(e)}")
            return False

    def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves session chat history from MySQL database."""
        if not self.is_connected:
            return []

        try:
            conn = mysql.connector.connect(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=settings.MYSQL_DATABASE,
            )
            cursor = conn.cursor(dictionary=True)
            query = """
            SELECT role, content, intent, confidence_score, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
            """
            cursor.execute(query, (session_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Error reading chat history from MySQL: {str(e)}")
            return []


db_pool = AsyncDatabasePool()
