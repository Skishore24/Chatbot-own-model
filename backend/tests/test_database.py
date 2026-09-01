"""
backend/tests/test_database.py
----------------------------------------------------
Tests for Database Manager and Repository persistence.
"""

import pytest
from app.database.connection import db_manager
from app.database.repository import ChatRepository, LeadRepository


def test_database_manager_state():
    # Manager should be initialized with active engine
    assert db_manager.is_available is True
    assert db_manager.engine_type in ["mysql", "sqlite"]


def test_repository_save_and_retrieve_session():
    session_id = "test_db_session_123"
    msg_id = ChatRepository.save_message(
        session_id=session_id,
        role_or_sender="user",
        content="Testing DB persistence",
    )
    assert msg_id is not None
    assert msg_id >= 0

    history = ChatRepository.get_session_history(session_id)
    assert len(history) >= 1
    assert history[0]["text"] == "Testing DB persistence"


def test_lead_repository():
    lead_id = LeadRepository.create_lead(
        name="Test Client",
        email="testclient@example.com",
        company="Tech Corp",
        message="Automated unit test lead",
    )
    assert lead_id is not None
    assert lead_id >= 0
