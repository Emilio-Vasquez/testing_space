from __future__ import annotations

from typing import Dict

AttackState = Dict[str, object]
ContainmentState = Dict[str, object]
SessionState = Dict[str, object]

ACTIVE_ATTACKS: Dict[str, AttackState] = {}
CONTAINMENT_STATES: Dict[str, ContainmentState] = {}
ATTACK_SESSIONS: Dict[str, SessionState] = {}
