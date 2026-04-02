from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .attacks import apply_attack
from .containment import attack_signature, get_containment_state, summarize_containment
from .detection import detect_anomalies
from .imaging import compute_histogram, compute_valid_crop_bounds, crop_with_bounds, load_science_image, normalize_to_uint8
from .state import ACTIVE_ATTACKS


def prepare_raw_view(slug: str) -> Optional[Dict[str, object]]:
    payload = load_science_image(slug)
    if payload is None:
        return None
    clean_arr, meta = payload
    attack = ACTIVE_ATTACKS.get(slug); attack_active = attack is not None; state = get_containment_state(slug)
    attacked_arr = apply_attack(clean_arr, attack) if attack_active else clean_arr.copy()
    anomaly_mask, anomaly = detect_anomalies(clean_arr, attacked_arr)
    current_attack_signature = attack_signature(attack)
    if attack_active and anomaly.get("severity") in {"medium", "high"} and not state.get("contained") and state.get("suppressed_attack_signature") != current_attack_signature:
        state.update({"contained": True, "mode": "automatic", "display_mode": "attacked", "message": "Automatically contained due to anomaly severity."})
    display_base = clean_arr if state.get("display_mode") == "clean" else attacked_arr
    display_mask_source = np.zeros(clean_arr.shape, dtype=bool) if state.get("display_mode") == "clean" else anomaly_mask
    display_anomaly = dict(anomaly)
    if state.get("display_mode") == "clean":
        display_anomaly = {**display_anomaly, "overlay": None, "message": "Trusted baseline restored for inspection after analyst action."}
    crop_bounds = compute_valid_crop_bounds(clean_arr)
    display_source = crop_with_bounds(display_base, crop_bounds); display_mask = crop_with_bounds(display_mask_source, crop_bounds); display_arr = normalize_to_uint8(display_source)
    display_overlay = display_anomaly.get("overlay")
    if crop_bounds is not None and display_overlay is not None:
        r0, _r1, c0, _c1 = crop_bounds; bbox = display_overlay.get("bbox"); centroid = display_overlay.get("centroid")
        if bbox:
            bbox["x"] = int(bbox["x"] - c0); bbox["y"] = int(bbox["y"] - r0)
        if centroid:
            centroid["x"] = int(centroid["x"] - c0); centroid["y"] = int(centroid["y"] - r0)
    containment = summarize_containment(slug, anomaly, attack_active)
    cropped_meta = {**meta, "display_shape": list(display_arr.shape), "crop_bounds": {"row_start": crop_bounds[0], "row_end": crop_bounds[1], "col_start": crop_bounds[2], "col_end": crop_bounds[3]} if crop_bounds is not None else None}
    return {"clean": clean_arr, "attacked": attacked_arr, "display": display_arr, "display_mask": display_mask, "anomaly_mask": anomaly_mask, "anomaly": display_anomaly, "raw_anomaly": anomaly, "attack_active": attack_active, "attack": attack, "containment": containment, "meta": cropped_meta}


def compute_stats_payload(view: Dict[str, object]) -> Dict[str, object]:
    display = np.asarray(view["display"], dtype=np.uint8); display_mask = np.asarray(view["display_mask"], dtype=bool)
    flat = display.astype(np.float32).ravel(); anomaly_flat = display[display_mask].astype(np.float32) if display_mask.any() else np.array([], dtype=np.float32)
    return {"min": float(flat.min()) if flat.size else None, "max": float(flat.max()) if flat.size else None, "mean": float(flat.mean()) if flat.size else None, "median": float(np.median(flat)) if flat.size else None, "std": float(flat.std()) if flat.size else None, "histogram": compute_histogram(flat), "anomaly_histogram": compute_histogram(anomaly_flat) if anomaly_flat.size else [0] * 20, "attack_active": view["attack_active"], "attack": view["attack"], "anomaly": view["anomaly"], "containment": view["containment"], "meta": view["meta"]}
