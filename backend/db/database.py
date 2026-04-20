import sqlite3
from app.config import DB_PATH, logger


# ─────────────────────────────────────────────
# CONNECTION (OPTIMIZED)
# ─────────────────────────────────────────────
def get_connection():
    """Returns optimized SQLite connection (production-ready)."""
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # 🔥 Performance optimizations
    conn.execute("PRAGMA journal_mode=WAL;")   # prevents locking
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    return conn


# ─────────────────────────────────────────────
# INIT DATABASE
# ─────────────────────────────────────────────
def init_db():
    """Initializes all database tables and indexes."""

    try:
        with get_connection() as conn:
            cur = conn.cursor()

            # ───────── CHATS ─────────
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT,
                answer TEXT,
                intent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # ───────── LEADS ─────────
            cur.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # ───────── FEEDBACK (IMPORTANT FOR AI TRAINING) ─────────
            cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # ───────── INDEXES (CRITICAL FOR PERFORMANCE) ─────────
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_session ON chats(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_time ON chats(created_at)")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_time ON leads(created_at)")

            conn.commit()

            logger.info("✅ Database initialized (Production Ready)")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise