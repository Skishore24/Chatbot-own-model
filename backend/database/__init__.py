"""
database/__init__.py
----------------------------------------------------
Re-exports all public database functions so the rest
of the codebase can simply do:

    from database import init_db, save_chat_to_db, ...
"""

from database.mysql import (
    init_db,
    save_chat_to_db,
    save_lead_to_db,
    save_feedback_to_db,
    update_user_profile,
    get_user_profile,
    increment_chat_count,
    get_connection,
    get_chat_history,
)

__all__ = [
    "init_db",
    "save_chat_to_db",
    "save_lead_to_db",
    "save_feedback_to_db",
    "update_user_profile",
    "get_user_profile",
    "increment_chat_count",
    "get_connection",
    "get_chat_history",
]

