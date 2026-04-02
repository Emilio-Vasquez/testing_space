from __future__ import annotations

from flask import Blueprint, jsonify, request

from dashboard.pipeline.attacks import validate_attack_payload
from dashboard.pipeline.catalog import get_raw_objects
from dashboard.pipeline.containment import attack_signature, get_containment_state, reset_containment_state, summarize_containment
from dashboard.pipeline.sessions import disconnect_sessions_for_slug, get_attack_session, start_attack_session
from dashboard.pipeline.state import ACTIVE_ATTACKS
from dashboard.pipeline.view_service import prepare_raw_view

attack_api_bp = Blueprint("attack_api_bp", __name__)


@attack_api_bp.route("/api/session_start", methods=["POST"])
def api_session_start():
    payload = request.get_json(silent=True) or {}
    slug = str(payload.get("slug", "")).strip().lower()
    client_name = str(payload.get("client_name", "raspberry-pi")).strip() or "raspberry-pi"
    if not slug:
        return jsonify({"error": "Missing 'slug'."}), 400
    if slug not in get_raw_objects():
        return jsonify({"error": "Unknown slug."}), 404
    state = get_containment_state(slug)
    if state.get("disconnected"):
        return jsonify({"error": "Attack source currently disconnected for this dataset.", "slug": slug, "containment": summarize_containment(slug, {"severity": "none"}, slug in ACTIVE_ATTACKS)}), 423
    session = start_attack_session(slug, client_name)
    return jsonify({"message": "Attack session started.", "session_id": session["session_id"], "slug": slug, "client_name": client_name})


@attack_api_bp.route("/api/session_status/<session_id>")
def api_session_status(session_id: str):
    session = get_attack_session(session_id.strip())
    if session is None:
        return jsonify({"error": "Unknown session_id."}), 404
    return jsonify({"session_id": session["session_id"], "slug": session["slug"], "client_name": session["client_name"], "active": bool(session.get("active", False)), "disconnect_reason": session.get("disconnect_reason")})


@attack_api_bp.route("/api/attack_ingest", methods=["POST"])
def api_attack_ingest():
    payload = request.get_json(silent=True) or {}
    ok, message = validate_attack_payload(payload)
    if not ok:
        return jsonify({"error": message}), 400
    slug = str(payload.get("slug", "")).strip().lower(); session_id = str(payload.get("session_id", "")).strip() or None; state = get_containment_state(slug)
    if session_id:
        session = get_attack_session(session_id)
        if session is None:
            return jsonify({"error": "Unknown session_id."}), 404
        if not session.get("active", False):
            return jsonify({"message": "Attack session inactive. Payload rejected.", "slug": slug, "session_id": session_id, "blocked": True, "disconnect_reason": session.get("disconnect_reason")}), 423
    if state.get("disconnected"):
        state["blocked_attempts"] = int(state.get("blocked_attempts", 0) or 0) + 1
        if session_id and get_attack_session(session_id) is not None:
            get_attack_session(session_id)["active"] = False; get_attack_session(session_id)["disconnect_reason"] = "disconnected_by_defender"
        return jsonify({"message": "Attack source disconnected. Incoming attack blocked.", "slug": slug, "session_id": session_id, "blocked": True, "containment": summarize_containment(slug, {"severity": "none"}, slug in ACTIVE_ATTACKS)}), 423
    ACTIVE_ATTACKS[slug] = {**payload, "slug": slug, "attack_type": str(payload.get("attack_type", "")).strip().lower()}
    state["suppressed_attack_signature"] = None
    view = prepare_raw_view(slug)
    return jsonify({"message": "Attack accepted and applied to the raw visualization layer.", "slug": slug, "session_id": session_id, "attack": ACTIVE_ATTACKS[slug], "anomaly": view["anomaly"] if view else None, "containment": view["containment"] if view else None})


@attack_api_bp.route("/api/contain/<slug>", methods=["POST"])
def api_contain(slug: str):
    slug = slug.strip().lower()
    if slug not in get_raw_objects():
        return jsonify({"error": "Unknown raw dataset"}), 404
    state = get_containment_state(slug)
    state.update({"contained": True, "mode": "manual", "display_mode": "attacked", "message": "Manual containment enabled for analyst inspection."})
    view = prepare_raw_view(slug)
    return jsonify({"message": "Manual containment enabled.", "slug": slug, "containment": view["containment"] if view else state})


@attack_api_bp.route("/api/disconnect_attacker/<slug>", methods=["POST"])
def api_disconnect_attacker(slug: str):
    slug = slug.strip().lower()
    if slug not in get_raw_objects():
        return jsonify({"error": "Unknown raw dataset"}), 404
    ACTIVE_ATTACKS.pop(slug, None)
    disconnected_sessions = disconnect_sessions_for_slug(slug)
    state = get_containment_state(slug)
    state.update({"contained": True, "mode": state.get("mode") or "manual", "display_mode": "clean", "disconnected": True, "message": "Suspected attack source disconnected. Trusted baseline restored."})
    view = prepare_raw_view(slug)
    return jsonify({"message": "Attacker disconnected and trusted baseline restored.", "slug": slug, "terminated_sessions": disconnected_sessions, "containment": view["containment"] if view else state})


@attack_api_bp.route("/api/release/<slug>", methods=["POST"])
def api_release(slug: str):
    slug = slug.strip().lower()
    if slug not in get_raw_objects():
        return jsonify({"error": "Unknown raw dataset"}), 404
    state = reset_containment_state(slug)
    if slug in ACTIVE_ATTACKS:
        state["suppressed_attack_signature"] = attack_signature(ACTIVE_ATTACKS.get(slug))
    view = prepare_raw_view(slug)
    return jsonify({"message": "Containment released. Monitoring resumed.", "slug": slug, "containment": view["containment"] if view else state})


@attack_api_bp.route("/api/attack_clear/<slug>", methods=["POST"])
def api_attack_clear(slug: str):
    slug = slug.strip().lower(); removed = ACTIVE_ATTACKS.pop(slug, None); state = get_containment_state(slug)
    if not state.get("disconnected"):
        state["display_mode"] = "attacked"
    state["suppressed_attack_signature"] = None
    return jsonify({"message": "Attack state cleared.", "slug": slug, "cleared": removed is not None})
