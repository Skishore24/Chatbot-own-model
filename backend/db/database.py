import sqlite3
from app.config import DB_PATH, logger

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes all database tables and indexes for production use."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()

            # 1. CHATS TABLE
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT,
                answer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 2. LEADS TABLE
            cur.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 3. FEEDBACK TABLE
            cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 4. INDEXES (Performance)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_session ON chats(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")

            conn.commit()
            logger.info("✅ Database initialized (Chats, Leads, Feedback tables verified)")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise