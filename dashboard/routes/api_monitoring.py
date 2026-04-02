from __future__ import annotations

from flask import Blueprint, jsonify

from dashboard.pipeline.catalog import get_raw_objects
from dashboard.pipeline.view_service import compute_stats_payload, prepare_raw_view

monitoring_api_bp = Blueprint("monitoring_api_bp", __name__)


@monitoring_api_bp.route("/api/raw_data/<slug>")
def api_raw_data(slug: str):
    view = prepare_raw_view(slug)
    if view is None:
        return jsonify({"error": "Failed to parse FITS or no usable 2-D image found."}), 500
    arr = view["display"]
    return jsonify({"width": int(arr.shape[1]), "height": int(arr.shape[0]), "data": arr.tolist(), "meta": view["meta"], "attack_active": view["attack_active"], "attack": view["attack"], "anomaly": view["anomaly"], "containment": view["containment"]})


@monitoring_api_bp.route("/api/raw_stats/<slug>")
def api_raw_stats(slug: str):
    view = prepare_raw_view(slug)
    if view is None:
        return jsonify({"error": "Failed to parse FITS or no usable 2-D image found."}), 500
    return jsonify(compute_stats_payload(view))


@monitoring_api_bp.route("/api/attack_status/<slug>")
def api_attack_status(slug: str):
    if slug not in get_raw_objects():
        return jsonify({"error": "Unknown raw dataset"}), 404
    view = prepare_raw_view(slug)
    if view is None:
        return jsonify({"error": "Failed to load dataset"}), 500
    return jsonify({"slug": slug, "attack_active": view["attack_active"], "attack": view["attack"], "anomaly": view["anomaly"], "containment": view["containment"], "meta": view["meta"]})
