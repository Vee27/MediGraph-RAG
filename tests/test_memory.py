import pytest
from app.memory.session_memory import SessionMemory


def test_add_and_get_message():
    """Messages added should appear in history"""
    mem = SessionMemory()
    mem.add_message("session-1", "user", "What meds?")
    mem.add_message("session-1", "assistant", "Metformin 500mg")

    history = mem.get_history("session-1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What meds?"
    assert history[1]["role"] == "assistant"


def test_sessions_are_isolated():
    """Different session IDs should not share history"""
    mem = SessionMemory()
    mem.add_message("session-A", "user", "Message for A")
    mem.add_message("session-B", "user", "Message for B")

    history_a = mem.get_history("session-A")
    history_b = mem.get_history("session-B")

    assert len(history_a) == 1
    assert len(history_b) == 1
    assert history_a[0]["content"] == "Message for A"
    assert history_b[0]["content"] == "Message for B"


def test_clear_session():
    """Clearing a session should remove all its history"""
    mem = SessionMemory()
    mem.add_message("session-1", "user", "Hello")
    mem.clear_session("session-1")

    history = mem.get_history("session-1")
    assert len(history) == 0


def test_history_trimmed_to_max():
    """History should not exceed MAX_HISTORY_LENGTH"""
    from app.memory.session_memory import MAX_HISTORY_LENGTH
    mem = SessionMemory()

    # Add more messages than the max
    for i in range(MAX_HISTORY_LENGTH + 5):
        mem.add_message("session-1", "user", f"Message {i}")

    history = mem.get_history("session-1")
    assert len(history) <= MAX_HISTORY_LENGTH


def test_patient_id_stored():
    """patient_id should be associated with the session"""
    mem = SessionMemory()
    mem.add_message(
        "session-1", "user", "What meds?", patient_id="patient-001"
    )
    assert mem.get_patient_id("session-1") == "patient-001"


def test_history_has_no_timestamps():
    """
    Timestamps are internal — history returned to the LLM
    should only have role and content
    """
    mem = SessionMemory()
    mem.add_message("session-1", "user", "Hello")
    history = mem.get_history("session-1")

    assert "timestamp" not in history[0]
    assert "role" in history[0]
    assert "content" in history[0]


def test_empty_session_returns_empty_list():
    """Accessing a non-existent session should return []"""
    mem = SessionMemory()
    history = mem.get_history("session-that-never-existed")
    assert history == []


def test_session_stats():
    """Stats should reflect actual message count"""
    mem = SessionMemory()
    mem.add_message("session-1", "user", "Q1", patient_id="p-001")
    mem.add_message("session-1", "assistant", "A1")

    stats = mem.get_session_stats("session-1")
    assert stats["message_count"] == 2
    assert stats["patient_id"] == "p-001"
    assert stats["session_id"] == "session-1"


def test_list_active_sessions():
    """list_active_sessions should return all session IDs"""
    mem = SessionMemory()
    mem.add_message("session-X", "user", "Hello")
    mem.add_message("session-Y", "user", "Hello")

    sessions = mem.list_active_sessions()
    assert "session-X" in sessions
    assert "session-Y" in sessions