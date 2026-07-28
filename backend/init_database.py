"""
init_database.py
----------------------------------------------------
Genkit AI - Database Setup Utility

Usage:
    cd backend
    python init_database.py

Recreates / verifies the database `genkit_ai` and all required
tables, indices, and views in MySQL.
"""

import os
import sys

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import logger, MYSQL_DATABASE, MYSQL_HOST, MYSQL_PORT
from database.mysql import ensure_database_exists, init_db, database_health

def main():
    logger.info("=" * 70)
    logger.info("GENKIT AI - DATABASE INITIALIZATION & RE-CREATION")
    logger.info("=" * 70)
    logger.info(f"Target Database : {MYSQL_DATABASE}")
    logger.info(f"MySQL Host      : {MYSQL_HOST}:{MYSQL_PORT}")
    logger.info("=" * 70)

    try:
        logger.info("Step 1: Ensuring database exists...")
        ensure_database_exists()

        logger.info("Step 2: Initializing tables, indices, and views...")
        init_db()

        logger.info("Step 3: Checking database health...")
        health = database_health()

        if health.get("connected"):
            logger.info("=" * 70)
            logger.info("✓ DATABASE INITIALIZED SUCCESSFULLY!")
            logger.info(f"  Database : {health.get('database')}")
            logger.info(f"  Version  : {health.get('version')}")
            logger.info("=" * 70)
        else:
            logger.error(f"❌ Database health check failed: {health.get('error')}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Failed to initialize database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
