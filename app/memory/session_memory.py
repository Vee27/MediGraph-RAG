import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# How many messages to keep per session
# Older messages are dropped to avoid context window overflow
MAX_HISTORY_LENGTH = 10


class SessionMemory:
    """
    In-memory conversation store keyed by session_id.

    Structure:
        {
            "session-abc": {
                "messages": [
                    {"role": "user",      "content": "What meds?",    "timestamp": "..."},
                    {"role": "assistant", "content": "Metformin...",   "timestamp": "..."},
                    {"role": "user",      "content": "What dosage?",   "timestamp": "..."},
                    {"role": "assistant", "content": "500mg twice...", "timestamp": "..."},
                ],
                "patient_id": "patient-001",
                "created_at": "2024-01-15T09:30:00",
            }
        }

    Why in-memory and not a database?
        For Day 9 this is intentional — it's fast, simple,
        and good enough for a single-server setup.
        Data is lost on server restart, which is acceptable
        for development. Day 10+ could persist to SQLite.
    """

    def __init__(self):
        # defaultdict means accessing a missing key creates
        # an empty dict automatically instead of raising KeyError
        self._store: dict = defaultdict(lambda: {
            "messages": [],
            "patient_id": "",
            "created_at": datetime.now().isoformat(),
        })

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        patient_id: str = "",
    ) -> None:
        """
        Add one message to a session's history.

        Args:
            session_id: Unique conversation identifier
            role:       "user" or "assistant"
            content:    The message text
            patient_id: Associate a patient with this session
        """
        session = self._store[session_id]

        # Update patient_id if provided
        if patient_id:
            session["patient_id"] = patient_id

        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

        # Trim to max length — drop oldest messages first
        # Always keep pairs (user + assistant) so we don't
        # end up with an orphaned user message
        messages = session["messages"]
        if len(messages) > MAX_HISTORY_LENGTH:
            # Drop two at a time (one exchange) from the front
            session["messages"] = messages[-MAX_HISTORY_LENGTH:]

        logger.info(
            f"[memory] session={session_id} | "
            f"role={role} | "
            f"history_length={len(session['messages'])}"
        )

    def get_history(self, session_id: str) -> list[dict]:
        """
        Get the conversation history for a session.

        Returns:
            List of message dicts WITHOUT timestamps
            (timestamps are internal — Ollama doesn't need them)
            e.g. [{"role": "user", "content": "..."}, ...]
        """
        messages = self._store[session_id]["messages"]

        # Strip timestamps before returning — Ollama only
        # wants role and content
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

    def get_patient_id(self, session_id: str) -> str:
        """Get the patient_id associated with this session."""
        return self._store[session_id].get("patient_id", "")

    def clear_session(self, session_id: str) -> None:
        """
        Wipe a session's history.
        Useful when a physician starts a new patient case.
        """
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"[memory] Cleared session: {session_id}")

    def get_session_stats(self, session_id: str) -> dict:
        """Return stats about a session — useful for debugging."""
        session = self._store.get(session_id, {})
        messages = session.get("messages", [])
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "patient_id": session.get("patient_id", ""),
            "created_at": session.get("created_at", ""),
        }

    def list_active_sessions(self) -> list[str]:
        """Return all session IDs currently in memory."""
        return list(self._store.keys())


# Single shared instance used across all requests
session_memory = SessionMemory()