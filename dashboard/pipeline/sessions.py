from __future__ import annotations

from typing import Optional
from uuid import uuid4

from .state import ATTACK_SESSIONS, SessionState


def default_session_state(session_id: str, slug: str, client_name: str) -> SessionState:
    return {"session_id": session_id, "slug": slug, "client_name": client_name, "active": True, "disconnect_reason": None}


def start_attack_session(slug: str, client_name: str = "raspberry-pi") -> SessionState:
    session_id = uuid4().hex[:12]; session = default_session_state(session_id, slug, client_name); ATTACK_SESSIONS[session_id] = session; return session


def get_attack_session(session_id: str) -> Optional[SessionState]:
    return ATTACK_SESSIONS.get(session_id)


def disconnect_sessions_for_slug(slug: str, reason: str = "disconnected_by_defender") -> int:
    disconnected_count = 0
    for session in ATTACK_SESSIONS.values():
        if session.get("slug") == slug and session.get("active", False):
            session["active"] = False; session["disconnect_reason"] = reason; disconnected_count += 1
    return disconnected_count
