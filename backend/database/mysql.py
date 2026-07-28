"""
database/mysql.py
----------------------------------------------------
Genkit AI - MySQL Database Layer

Features
--------
• MySQL Connection Pool
• Automatic Reconnection
• Thread Safe
• Transaction Support
• CRUD Operations
• Chat History
• User Profiles
• Lead Management
• Feedback
• Analytics

Author : Genkit AI
"""

import os
import re
import sys
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Any

import mysql.connector
from mysql.connector import pooling, Error

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DATABASE,
    MYSQL_USER,
    MYSQL_PASSWORD,
    logger,
)

# ============================================================
# DATABASE CREATION BOOTSTRAP
# ============================================================

def ensure_database_exists():
    """
    Connect to MySQL server directly without specifying a database name,
    and create target database if it does not exist yet.
    """
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            autocommit=True,
        )
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        cursor.close()
        conn.close()
        logger.info(f"Verified/Created MySQL database: {MYSQL_DATABASE}")
    except Exception as e:
        logger.exception(f"Failed to verify or create MySQL database '{MYSQL_DATABASE}'")
        raise

# ============================================================
# CONNECTION POOL
# ============================================================

_POOL = None

POOL_SIZE = 10

POOL_NAME = "genkit_pool"


def initialize_pool():
    """
    Create MySQL connection pool.
    """

    global _POOL

    if _POOL is not None:
        return

    ensure_database_exists()

    try:

        _POOL = pooling.MySQLConnectionPool(

            pool_name=POOL_NAME,

            pool_size=POOL_SIZE,

            pool_reset_session=True,

            host=MYSQL_HOST,

            port=MYSQL_PORT,

            user=MYSQL_USER,

            password=MYSQL_PASSWORD,

            database=MYSQL_DATABASE,

            autocommit=False,

            charset="utf8mb4",

            collation="utf8mb4_unicode_ci"

        )

        logger.info(
            "MySQL connection pool initialized."
        )

    except Error:

        logger.exception(
            "Unable to initialize MySQL pool."
        )

        raise


# ============================================================
# CONNECTION
# ============================================================

def get_connection():
    """
    Return pooled MySQL connection.
    """

    if _POOL is None:

        initialize_pool()

    try:

        conn = _POOL.get_connection()

        if not conn.is_connected():

            conn.reconnect(
                attempts=3,
                delay=2
            )

        return conn

    except Error:

        logger.exception(
            "Failed to get MySQL connection."
        )

        raise


# ============================================================
# CURSOR CONTEXT
# ============================================================

@contextmanager
def get_cursor(dictionary=True):
    """
    Safe cursor manager.

    Automatically commits on success.

    Rolls back on failure.
    """

    conn = get_connection()

    cursor = conn.cursor(
        dictionary=dictionary
    )

    try:

        yield conn, cursor

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cursor.close()

        conn.close()

# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    """
    Create all required MySQL tables.

    Safe to call multiple times.
    """

    with get_cursor(dictionary=False) as (conn, cursor):

        # ====================================================
        # CHATS
        # ====================================================

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS chats (

            id BIGINT AUTO_INCREMENT PRIMARY KEY,

            session_id VARCHAR(120) NOT NULL,

            question LONGTEXT NOT NULL,

            answer LONGTEXT NOT NULL,

            intent VARCHAR(100) DEFAULT 'general',

            source VARCHAR(100) DEFAULT 'rag',

            confidence FLOAT DEFAULT 0,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            INDEX idx_session(session_id),

            INDEX idx_created(created_at)

        )

        """)

        # ====================================================
        # LEADS
        # ====================================================

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS leads (

            id BIGINT AUTO_INCREMENT PRIMARY KEY,

            name VARCHAR(150) NOT NULL,

            email VARCHAR(255) NOT NULL,

            phone VARCHAR(30),

            company VARCHAR(200),

            message TEXT,

            status VARCHAR(50) DEFAULT 'New',

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE KEY unique_email(email),

            INDEX idx_status(status)

        )

        """)

        # ====================================================
        # FEEDBACK
        # ====================================================

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS feedback (

            id BIGINT AUTO_INCREMENT PRIMARY KEY,

            session_id VARCHAR(120) NOT NULL,

            question LONGTEXT,

            answer LONGTEXT,

            rating INT NOT NULL,

            comments TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            INDEX idx_rating(rating),

            INDEX idx_session(session_id)

        )

        """)

        # ====================================================
        # USER PROFILES
        # ====================================================

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS user_profiles (

            id BIGINT AUTO_INCREMENT PRIMARY KEY,

            session_id VARCHAR(120) NOT NULL UNIQUE,

            name VARCHAR(150),

            email VARCHAR(255),

            phone VARCHAR(30),

            company VARCHAR(200),

            interest TEXT,

            last_query TEXT,

            total_chats INT DEFAULT 0,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP,

            INDEX idx_email(email),

            INDEX idx_name(name)

        )

        """)

        # ====================================================
        # DATABASE SCHEMA MIGRATIONS (Ensure all columns exist)
        # ====================================================
        migrations = [
            ("chats", "ADD COLUMN intent VARCHAR(100) DEFAULT 'general'"),
            ("chats", "ADD COLUMN source VARCHAR(100) DEFAULT 'rag'"),
            ("chats", "ADD COLUMN confidence FLOAT DEFAULT 0"),
            ("user_profiles", "ADD COLUMN phone VARCHAR(30)"),
            ("user_profiles", "ADD COLUMN company VARCHAR(200)"),
            ("user_profiles", "ADD COLUMN interest TEXT"),
            ("user_profiles", "ADD COLUMN last_query TEXT"),
            ("user_profiles", "ADD COLUMN total_chats INT DEFAULT 0"),
            ("feedback", "ADD COLUMN question LONGTEXT"),
            ("feedback", "ADD COLUMN answer LONGTEXT"),
        ]

        for table, action in migrations:
            try:
                cursor.execute(f"ALTER TABLE {table} {action}")
            except mysql.connector.Error as err:
                if err.errno != 1060:  # 1060 = Duplicate column name
                    logger.warning(f"Migration failed for {table} {action}: {err}")

        logger.info(
            "All MySQL tables verified successfully."
        )


# ============================================================
# DATABASE HEALTH
# ============================================================

def database_health() -> dict:
    """
    Check MySQL connection.
    """

    try:

        with get_cursor() as (conn, cursor):

            cursor.execute("SELECT VERSION() AS version")

            version = cursor.fetchone()["version"]

            return {

                "connected": True,

                "database": MYSQL_DATABASE,

                "version": version

            }

    except Exception as e:

        logger.exception(e)

        return {

            "connected": False,

            "database": MYSQL_DATABASE,

            "error": str(e)

        }
# ============================================================
# CHAT CRUD
# ============================================================

def save_chat_to_db(
    session_id: str,
    question: str,
    answer: str,
    intent: str = "general",
    source: str = "rag",
    confidence: float = 0.0,
):
    """
    Save chat into MySQL.
    """

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                INSERT INTO chats
                (
                    session_id,
                    question,
                    answer,
                    intent,
                    source,
                    confidence
                )
                VALUES
                (%s,%s,%s,%s,%s,%s)
                """,
                (
                    session_id,
                    question,
                    answer,
                    intent,
                    source,
                    confidence,
                ),
            )

        logger.info(
            f"Chat saved ({session_id})"
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# CHAT HISTORY
# ============================================================

def get_chat_history(
    session_id: str,
    limit: int = 20,
):
    """
    Load chat history from MySQL.
    """

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT
                    question,
                    answer,
                    intent,
                    source,
                    confidence,
                    created_at
                FROM chats
                WHERE session_id=%s
                ORDER BY id DESC
                LIMIT %s
                """,
                (
                    session_id,
                    limit,
                ),
            )

            rows = cursor.fetchall()

            rows.reverse()

            return rows

    except Exception as e:

        logger.exception(e)

        return []


# ============================================================
# LAST CHAT
# ============================================================

def get_last_chat(session_id: str):

    history = get_chat_history(
        session_id,
        limit=1,
    )

    if history:

        return history[0]

    return None


# ============================================================
# DELETE CHAT
# ============================================================

def delete_chat_history(session_id: str):

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                DELETE FROM chats
                WHERE session_id=%s
                """,
                (session_id,),
            )

        logger.info(
            f"Deleted chats ({session_id})"
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# CHAT COUNT
# ============================================================

def total_chats():

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                "SELECT COUNT(*) AS total FROM chats"
            )

            return cursor.fetchone()["total"]

    except Exception:

        return 0
# ============================================================
# LEAD CRUD
# ============================================================

def save_lead_to_db(
    name: str,
    email: str,
    phone: str = "",
    company: str = "",
    message: str = "",
):
    """
    Save a new lead into MySQL.
    Updates existing lead if email already exists.
    """

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                INSERT INTO leads
                (
                    name,
                    email,
                    phone,
                    company,
                    message
                )
                VALUES
                (%s,%s,%s,%s,%s)

                ON DUPLICATE KEY UPDATE

                    name=VALUES(name),
                    phone=VALUES(phone),
                    company=VALUES(company),
                    message=VALUES(message)
                """,
                (
                    name,
                    email,
                    phone,
                    company,
                    message,
                ),
            )

        logger.info(f"Lead Saved : {email}")

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# GET LEADS
# ============================================================

def get_leads(limit: int = 100):

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT *
                FROM leads
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )

            return cursor.fetchall()

    except Exception as e:

        logger.exception(e)

        return []


# ============================================================
# GET LEAD BY EMAIL
# ============================================================

def get_lead(email: str):

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT *
                FROM leads
                WHERE email=%s
                LIMIT 1
                """,
                (email,),
            )

            return cursor.fetchone()

    except Exception:

        return None


# ============================================================
# UPDATE LEAD STATUS
# ============================================================

def update_lead_status(
    email: str,
    status: str,
):

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                UPDATE leads
                SET status=%s
                WHERE email=%s
                """,
                (
                    status,
                    email,
                ),
            )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# DELETE LEAD
# ============================================================

def delete_lead(email: str):

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                DELETE FROM leads
                WHERE email=%s
                """,
                (email,),
            )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# TOTAL LEADS
# ============================================================

def total_leads():

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT COUNT(*) AS total
                FROM leads
                """
            )

            return cursor.fetchone()["total"]

    except Exception:

        return 0

# ============================================================
# FEEDBACK CRUD
# ============================================================

def save_feedback_to_db(
    session_id: str,
    question: str,
    answer: str,
    rating: int,
    comments: str = "",
):
    """
    Save user feedback.
    """

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                INSERT INTO feedback
                (
                    session_id,
                    question,
                    answer,
                    rating,
                    comments
                )
                VALUES
                (%s,%s,%s,%s,%s)
                """,
                (
                    session_id,
                    question,
                    answer,
                    rating,
                    comments,
                ),
            )

        logger.info(
            f"Feedback saved ({session_id})"
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# GET FEEDBACK
# ============================================================

def get_feedback(limit: int = 100):

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT *
                FROM feedback
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )

            return cursor.fetchall()

    except Exception as e:

        logger.exception(e)

        return []


# ============================================================
# DELETE FEEDBACK
# ============================================================

def delete_feedback(feedback_id: int):

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                DELETE FROM feedback
                WHERE id=%s
                """,
                (feedback_id,),
            )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# AVERAGE RATING
# ============================================================

def average_rating():

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT
                    ROUND(AVG(rating),2) AS avg_rating
                FROM feedback
                """
            )

            row = cursor.fetchone()

            if row and row["avg_rating"] is not None:
                return float(row["avg_rating"])

            return 0.0

    except Exception:

        return 0.0


# ============================================================
# TOTAL FEEDBACK
# ============================================================

def total_feedback():

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT COUNT(*) AS total
                FROM feedback
                """
            )

            row = cursor.fetchone()

            return row["total"]

    except Exception:

        return 0

# ============================================================
# USER PROFILE CRUD
# ============================================================

def update_user_profile(
    session_id: str,
    name: str = "",
    email: str = "",
    phone: str = "",
    company: str = "",
    interest: str = "",
    last_query: str = "",
):
    """
    Create or update a user profile.
    """

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                INSERT INTO user_profiles
                (
                    session_id,
                    name,
                    email,
                    phone,
                    company,
                    interest,
                    last_query
                )
                VALUES
                (%s,%s,%s,%s,%s,%s,%s)

                ON DUPLICATE KEY UPDATE

                    name=VALUES(name),
                    email=VALUES(email),
                    phone=VALUES(phone),
                    company=VALUES(company),
                    interest=VALUES(interest),
                    last_query=VALUES(last_query)
                """,
                (
                    session_id,
                    name,
                    email,
                    phone,
                    company,
                    interest,
                    last_query,
                ),
            )

        logger.info(
            f"User profile updated ({session_id})"
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# GET USER PROFILE
# ============================================================

def get_user_profile(session_id: str):

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT *
                FROM user_profiles
                WHERE session_id=%s
                LIMIT 1
                """,
                (session_id,),
            )

            return cursor.fetchone()

    except Exception as e:

        logger.exception(e)

        return None


# ============================================================
# INCREMENT CHAT COUNT
# ============================================================

def increment_chat_count(session_id: str):

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                UPDATE user_profiles
                SET total_chats = total_chats + 1
                WHERE session_id=%s
                """,
                (session_id,),
            )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# DELETE USER PROFILE
# ============================================================

def delete_user_profile(session_id: str):

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                DELETE FROM user_profiles
                WHERE session_id=%s
                """,
                (session_id,),
            )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# TOTAL USERS
# ============================================================

def total_users():

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT COUNT(*) AS total
                FROM user_profiles
                """
            )

            row = cursor.fetchone()

            return row["total"]

    except Exception:

        return 0
# ============================================================
# CONVERSATION HISTORY
# ============================================================

def get_conversation(
    session_id: str,
    limit: int = 20,
):
    """
    Return conversation in ChatGPT format.
    """

    history = get_chat_history(
        session_id=session_id,
        limit=limit,
    )

    conversation = []

    for row in history:

        conversation.append({

            "role": "user",

            "message": row["question"]

        })

        conversation.append({

            "role": "assistant",

            "message": row["answer"]

        })

    return conversation


# ============================================================
# RECENT CHATS
# ============================================================

def get_recent_chats(limit: int = 50):

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT *
                FROM chats
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )

            return cursor.fetchall()

    except Exception as e:

        logger.exception(e)

        return []


# ============================================================
# SEARCH CHATS
# ============================================================

def search_chats(keyword: str):

    try:

        keyword = f"%{keyword}%"

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT *
                FROM chats
                WHERE question LIKE %s
                   OR answer LIKE %s
                ORDER BY created_at DESC
                """,
                (
                    keyword,
                    keyword,
                ),
            )

            return cursor.fetchall()

    except Exception as e:

        logger.exception(e)

        return []


# ============================================================
# SESSION EXISTS
# ============================================================

def session_exists(session_id: str):

    try:

        with get_cursor() as (_, cursor):

            cursor.execute(
                """
                SELECT COUNT(*) AS total
                FROM chats
                WHERE session_id=%s
                """,
                (session_id,),
            )

            row = cursor.fetchone()

            return row["total"] > 0

    except Exception:

        return False


# ============================================================
# DELETE SESSION
# ============================================================

def delete_session(session_id: str):

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute(
                """
                DELETE FROM chats
                WHERE session_id=%s
                """,
                (session_id,),
            )

            cursor.execute(
                """
                DELETE FROM feedback
                WHERE session_id=%s
                """,
                (session_id,),
            )

            cursor.execute(
                """
                DELETE FROM user_profiles
                WHERE session_id=%s
                """,
                (session_id,),
            )

        logger.info(
            f"Session deleted ({session_id})"
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False

# ============================================================
# DATABASE ANALYTICS
# ============================================================

def database_statistics():
    """
    Return dashboard statistics.
    """

    return {

        "users": total_users(),

        "chats": total_chats(),

        "leads": total_leads(),

        "feedback": total_feedback(),

        "average_rating": average_rating(),

    }


# ============================================================
# DATABASE CLEANUP
# ============================================================

def optimize_database():
    """
    Optimize all tables.
    """

    tables = [

        "chats",

        "leads",

        "feedback",

        "user_profiles",

    ]

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            for table in tables:

                cursor.execute(
                    f"OPTIMIZE TABLE {table}"
                )

        logger.info(
            "Database optimized successfully."
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# DATABASE RESET
# ============================================================

def clear_database():
    """
    Delete all chatbot data.
    """

    try:

        with get_cursor(dictionary=False) as (_, cursor):

            cursor.execute("DELETE FROM chats")

            cursor.execute("DELETE FROM leads")

            cursor.execute("DELETE FROM feedback")

            cursor.execute("DELETE FROM user_profiles")

        logger.warning(
            "Database cleared."
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False


# ============================================================
# INITIALIZE DATABASE
# ============================================================

try:

    initialize_pool()

    init_db()

    logger.info(
        "Genkit MySQL initialized successfully."
    )

except Exception:

    logger.exception(
        "MySQL initialization failed."
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [

    # Connection

    "get_connection",

    "get_cursor",

    "initialize_pool",

    "init_db",

    "database_health",

    # Chats

    "save_chat_to_db",

    "get_chat_history",

    "get_last_chat",

    "get_recent_chats",

    "search_chats",

    "delete_chat_history",

    "delete_session",

    "get_conversation",

    "session_exists",

    "total_chats",

    # Leads

    "save_lead_to_db",

    "get_leads",

    "get_lead",

    "update_lead_status",

    "delete_lead",

    "total_leads",

    # Feedback

    "save_feedback_to_db",

    "get_feedback",

    "delete_feedback",

    "average_rating",

    "total_feedback",

    # Users

    "update_user_profile",

    "get_user_profile",

    "increment_chat_count",

    "delete_user_profile",

    "total_users",

    # Utilities

    "database_statistics",

    "optimize_database",

    "clear_database",

]
