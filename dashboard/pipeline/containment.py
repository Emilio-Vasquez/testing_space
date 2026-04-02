from __future__ import annotations

from typing import Dict, Optional

from .state import CONTAINMENT_STATES, AttackState, ContainmentState


def default_containment_state() -> ContainmentState:
    return {"contained": False, "mode": None, "display_mode": "attacked", "disconnected": False, "blocked_attempts": 0, "message": "Monitoring normally.", "suppressed_attack_signature": None}


def attack_signature(attack: Optional[AttackState]) -> Optional[str]:
    if attack is None:
        return None
    return str(tuple(sorted((str(k), repr(v)) for k, v in attack.items())))


def get_containment_state(slug: str) -> ContainmentState:
    state = CONTAINMENT_STATES.get(slug)
    if state is None:
        state = default_containment_state(); CONTAINMENT_STATES[slug] = state
    return state


def reset_containment_state(slug: str) -> ContainmentState:
    state = default_containment_state(); CONTAINMENT_STATES[slug] = state; return state


def summarize_containment(slug: str, anomaly: Dict[str, object], attack_active: bool) -> Dict[str, object]:
    state = get_containment_state(slug)
    contained = bool(state.get("contained", False)); mode = state.get("mode"); disconnected = bool(state.get("disconnected", False))
    display_mode = str(state.get("display_mode", "attacked")); blocked_attempts = int(state.get("blocked_attempts", 0) or 0)
    if contained and disconnected and display_mode == "clean":
        message = "Analyst containment active. Suspected source disconnected and trusted baseline restored for inspection."
    elif contained and mode == "manual":
        message = "Manual containment active. Frame held for analyst inspection. Use Resume Monitoring to continue watching the live attack state."
    elif contained and mode == "automatic":
        message = "Automatic containment active due to anomaly severity. Frame frozen pending analyst review."
    elif attack_active and anomaly.get("severity") in {"low", "medium", "high"}:
        message = "Attack activity present. Monitoring remains active."
    else:
        message = "Monitoring normally."
    if blocked_attempts > 0 and disconnected:
        message += f" Blocked attack attempts since disconnect: {blocked_attempts}."
    return {"contained": contained, "mode": mode, "disconnected": disconnected, "display_mode": display_mode, "blocked_attempts": blocked_attempts, "message": message, "allow_manual_contain": not contained, "allow_disconnect": contained or attack_active, "allow_resume": contained or disconnected}
