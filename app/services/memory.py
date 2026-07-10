"""
Session-based conversation memory.

In-memory store (per-process) mapping session_id -> Session. Good enough for
a single-instance deployment (which is what this assignment targets); a
production multi-instance deployment would swap this for Redis without
changing the public functions below.

Also handles auto-naming a session from its first question, purely via
simple truncation (no extra LLM call needed — keeps this fast and free).
"""

import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.core.config import settings
from app.models.schemas import ConversationTurn, Session

logger = logging.getLogger(__name__)

_sessions: Dict[str, Session] = {}


def create_session() -> Session:
    """Create and register a new empty session.

    Returns:
        The newly created Session.
    """
    session_id = str(uuid.uuid4())
    session = Session(session_id=session_id)
    _sessions[session_id] = session
    return session


def get_or_create_session(session_id: Optional[str]) -> Session:
    """Fetch an existing session, or create a new one if session_id is
    missing, unknown, or expired.

    Args:
        session_id: Client-provided session ID, or None to start fresh.

    Returns:
        A valid, non-expired Session.
    """
    _evict_expired_sessions()

    if session_id and session_id in _sessions:
        return _sessions[session_id]

    if session_id:
        logger.info(f"Unknown or expired session_id '{session_id}' — starting a new session.")

    return create_session()


def add_turn(session: Session, question: str, answer: str) -> None:
    """Append a Q&A turn to a session, trimming to the configured max length,
    and auto-generate a title if this is the first turn.

    Args:
        session: The session to update.
        question: The user's question.
        answer: The generated answer.
    """
    session.turns.append(ConversationTurn(question=question, answer=answer, timestamp=datetime.utcnow()))
    session.last_active_at = datetime.utcnow()

    if len(session.turns) > settings.MAX_CONVERSATION_TURNS:
        session.turns = session.turns[-settings.MAX_CONVERSATION_TURNS:]

    if session.title is None:
        session.title = _generate_title(question)


def get_recent_history(session: Session, max_turns: int = 2) -> List[ConversationTurn]:
    """Return the most recent N turns for inclusion in the LLM prompt.

    Args:
        session: The session to read from.
        max_turns: Number of most recent turns to return (keep small —
            only enough for resolving follow-up references like "what about X").

    Returns:
        List of ConversationTurn, oldest first.
    """
    return session.turns[-max_turns:] if session.turns else []


def _generate_title(question: str) -> str:
    """Auto-generate a short session title from the first question.

    Purely rule-based (no LLM call) to keep this instant and free.

    Args:
        question: The first question asked in the session.

    Returns:
        A short title string, e.g. "Annual leave entitlement".
    """
    cleaned = re.sub(r"[?!.]+$", "", question.strip())
    words = cleaned.split()
    title = " ".join(words[: settings.SESSION_NAME_MAX_WORDS])
    return title[0].upper() + title[1:] if title else "New conversation"


def _evict_expired_sessions() -> None:
    """Remove sessions that have been inactive longer than SESSION_TTL_SECONDS.

    Called opportunistically on each get_or_create_session call rather than
    via a background thread — simple and sufficient for this scale.
    """
    cutoff = datetime.utcnow() - timedelta(seconds=settings.SESSION_TTL_SECONDS)
    expired = [sid for sid, s in _sessions.items() if s.last_active_at < cutoff]
    for sid in expired:
        del _sessions[sid]
    if expired:
        logger.info(f"Evicted {len(expired)} expired session(s).")