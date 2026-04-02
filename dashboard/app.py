from __future__ import annotations

"""
Flask application for the Astrophysics Cybersecurity dashboard.

This version:
- reads real FITS/JWST/HST image data with Astropy
- keeps the baseline scientific frame immutable
- accepts Raspberry Pi attack payloads via a POST endpoint
- applies attacks only to an in-memory copy of the raw frame
- detects anomalies by comparing attacked data to the clean baseline
- returns image annotations and histogram overlays for the raw page
- crops display-only views to the valid data footprint for cleaner raw visualization
- supports automatic and manual containment workflows for analyst response
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from astropy.io import fits
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

ImageArray = np.ndarray
AttackState = Dict[str, object]
ContainmentState = Dict[str, object]
SessionState = Dict[str, object]

# In-memory store of active attacks keyed by slug.
ACTIVE_ATTACKS: Dict[str, AttackState] = {}
# In-memory store of containment state keyed by slug.
CONTAINMENT_STATES: Dict[str, ContainmentState] = {}
# In-memory store of Raspberry Pi attack sessions keyed by session id.
ATTACK_SESSIONS: Dict[str, SessionState] = {}


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    current_dir = Path(__file__).resolve().parent
    base_candidates = [
        current_dir / "data",
        current_dir.parents[1] / "data" if len(current_dir.parents) > 1 else None,
        current_dir.parents[2] / "data" if len(current_dir.parents) > 2 else None,
    ]
    base_candidates = [candidate for candidate in base_candidates if candidate is not None]

    data_base: Optional[Path] = None
    for candidate in base_candidates:
        if (candidate / "fits" / "mastDownload").exists():
            data_base = candidate
            break

    if data_base is None:
        raise RuntimeError("Cannot locate data directory. Ensure data/fits/mastDownload exists.")

    fits_base = data_base / "fits" / "mastDownload"
    rendered_base = data_base / "rendered"

    SLUG_TITLE_OVERRIDES: Dict[str, str] = {
        "ice00ap1q_raw": "Pillars of Creation (M16)",
        "hst_8063_p7_nic_nic3_f110w_01_drz": "Stephan’s Quintet",
        "jw02736006001_02101_00001_nrs1_cal": "SMACS 0723 Deep Field (JWST)",
    }

    PREFERRED_NAV_ORDER = [
        "hst_8063_p7_nic_nic3_f110w_01_drz",
        "jw02736006001_02101_00001_nrs1_cal",
        "ice00ap1q_raw",
    ]

    def discover_raw_files() -> Dict[str, Tuple[str, Path]]:
        mapping: Dict[str, Tuple[str, Path]] = {}
        if not fits_base.exists():
            return mapping

        paths: List[Path] = []
        for pattern in ("*.fits", "*.fit", "*.fts"):
            paths.extend(fits_base.rglob(pattern))

        for file in sorted(paths):
            if not file.is_file():
                continue
            slug = file.stem.replace(" ", "_").lower()
            title = SLUG_TITLE_OVERRIDES.get(slug, slug.replace("_", " ").title())
            mapping[slug] = (title, file)
        return mapping

    RAW_OBJECTS: Dict[str, Tuple[str, Path]] = discover_raw_files()

    def build_nav() -> List[Dict[str, str]]:
        nav_items: List[Dict[str, str]] = []
        for slug in PREFERRED_NAV_ORDER:
            if slug in RAW_OBJECTS:
                title, _path = RAW_OBJECTS[slug]
                nav_items.append({"slug": slug, "title": title})

        for slug, (title, _path) in RAW_OBJECTS.items():
            if slug not in PREFERRED_NAV_ORDER:
                nav_items.append({"slug": slug, "title": title})
        return nav_items

    def _default_containment_state() -> ContainmentState:
        return {
            "contained": False,
            "mode": None,
            "display_mode": "attacked",
            "disconnected": False,
            "blocked_attempts": 0,
            "message": "Monitoring normally.",
            "suppressed_attack_signature": None,
        }

    def _attack_signature(attack: Optional[AttackState]) -> Optional[str]:
        if attack is None:
            return None
        return str(tuple(sorted((str(k), repr(v)) for k, v in attack.items())))

    def _get_containment_state(slug: str) -> ContainmentState:
        state = CONTAINMENT_STATES.get(slug)
        if state is None:
            state = _default_containment_state()
            CONTAINMENT_STATES[slug] = state
        return state

    def _reset_containment_state(slug: str) -> ContainmentState:
        state = _default_containment_state()
        CONTAINMENT_STATES[slug] = state
        return state

    def _default_session_state(session_id: str, slug: str, client_name: str) -> SessionState:
        return {
            "session_id": session_id,
            "slug": slug,
            "client_name": client_name,
            "active": True,
            "disconnect_reason": None,
        }

    def _start_attack_session(slug: str, client_name: str = "raspberry-pi") -> SessionState:
        session_id = uuid4().hex[:12]
        session = _default_session_state(session_id, slug, client_name)
        ATTACK_SESSIONS[session_id] = session
        return session

    def _get_attack_session(session_id: str) -> Optional[SessionState]:
        return ATTACK_SESSIONS.get(session_id)

    def _disconnect_sessions_for_slug(slug: str, reason: str = "disconnected_by_defender") -> int:
        disconnected_count = 0
        for session in ATTACK_SESSIONS.values():
            if session.get("slug") == slug and session.get("active", False):
                session["active"] = False
                session["disconnect_reason"] = reason
                disconnected_count += 1
        return disconnected_count

    def _find_best_image_hdu(hdul: fits.HDUList) -> Optional[Tuple[int, str, ImageArray]]:
        def normalize_hdu_data(data: object) -> Optional[ImageArray]:
            if data is None:
                return None
            arr = np.asarray(data)
            if arr.size == 0:
                return None
            if arr.ndim == 2:
                return arr
            while arr.ndim > 2:
                arr = arr[0]
            return arr if arr.ndim == 2 else None

        for idx, hdu in enumerate(hdul):
            extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            arr = normalize_hdu_data(getattr(hdu, "data", None))
            if extname == "SCI" and arr is not None:
                return idx, extname or f"HDU{idx}", arr

        for idx, hdu in enumerate(hdul):
            extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            arr = normalize_hdu_data(getattr(hdu, "data", None))
            if arr is not None:
                return idx, extname or f"HDU{idx}", arr
        return None

    @lru_cache(maxsize=32)
    def _load_science_image(slug: str) -> Optional[Tuple[ImageArray, Dict[str, object]]]:
        entry = RAW_OBJECTS.get(slug)
        if entry is None:
            return None

        _title, path = entry
        try:
            with fits.open(path, memmap=False) as hdul:
                result = _find_best_image_hdu(hdul)
                if result is None:
                    return None
                hdu_index, hdu_name, raw_arr = result
                arr = np.asarray(raw_arr, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
                meta = {
                    "source_file": path.name,
                    "selected_hdu_index": hdu_index,
                    "selected_hdu_name": hdu_name,
                    "shape": list(arr.shape),
                    "dtype": str(raw_arr.dtype),
                }
                return arr.copy(), meta
        except Exception as exc:
            print(f"[ERROR] Failed to read FITS for slug '{slug}': {exc}")
            return None

    def _compute_valid_crop_bounds(arr: ImageArray, eps: float = 1e-8) -> Optional[Tuple[int, int, int, int]]:
        arr = np.asarray(arr, dtype=np.float32)
        valid_mask = np.isfinite(arr) & (np.abs(arr) > eps)

        if not valid_mask.any():
            return None

        rows = np.where(valid_mask.any(axis=1))[0]
        cols = np.where(valid_mask.any(axis=0))[0]

        if rows.size == 0 or cols.size == 0:
            return None

        r0, r1 = int(rows[0]), int(rows[-1] + 1)
        c0, c1 = int(cols[0]), int(cols[-1] + 1)
        return r0, r1, c0, c1

    def _crop_with_bounds(arr: ImageArray, bounds: Optional[Tuple[int, int, int, int]]) -> ImageArray:
        if bounds is None:
            return arr
        r0, r1, c0, c1 = bounds
        return arr[r0:r1, c0:c1]

    def _normalize_to_uint8(arr: ImageArray) -> ImageArray:
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)

        vmin, vmax = np.percentile(finite, [1.0, 99.8])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if vmax <= vmin:
                return np.zeros(arr.shape, dtype=np.uint8)

        clipped = np.clip(
            np.nan_to_num(arr, nan=vmin, posinf=vmax, neginf=vmin),
            vmin,
            vmax,
        )

        scaled = (clipped - vmin) / (vmax - vmin)
        stretched = np.arcsinh(12.0 * scaled) / np.arcsinh(12.0)

        return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)

    def _validate_attack_payload(payload: Dict[str, object]) -> Tuple[bool, str]:
        slug = str(payload.get("slug", "")).strip().lower()
        if not slug:
            return False, "Missing 'slug'."
        if slug not in RAW_OBJECTS:
            return False, "Unknown slug."

        attack_type = str(payload.get("attack_type", "")).strip().lower()
        allowed = {"hotspot", "stripe_noise", "block_dropout", "salt_pepper"}
        if attack_type not in allowed:
            return False, f"Unsupported attack_type. Allowed values: {sorted(allowed)}"

        session_id = str(payload.get("session_id", "")).strip()
        if session_id:
            session = _get_attack_session(session_id)
            if session is None:
                return False, "Unknown session_id."
            if str(session.get("slug", "")).strip().lower() != slug:
                return False, "session_id does not match slug."
            if not session.get("active", False):
                return False, "Session is no longer active."
        return True, "ok"

    def _apply_attack(clean: ImageArray, payload: AttackState) -> ImageArray:
        attacked = np.array(clean, copy=True, dtype=np.float32)
        attack_type = str(payload.get("attack_type", "")).lower()
        height, width = attacked.shape

        if attack_type == "hotspot":
            x = int(payload.get("x", width // 2))
            y = int(payload.get("y", height // 2))
            radius = max(1, int(payload.get("radius", 20)))
            intensity = float(payload.get("intensity", 4000.0))
            yy, xx = np.ogrid[:height, :width]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            attacked[mask] += intensity

        elif attack_type == "stripe_noise":
            axis = str(payload.get("axis", "vertical")).lower()
            start = int(payload.get("start", width // 2 if axis == "vertical" else height // 2))
            stripe_width = max(1, int(payload.get("width", 15)))
            intensity = float(payload.get("intensity", 1500.0))
            if axis == "horizontal":
                start = max(0, min(height - 1, start))
                end = max(start + 1, min(height, start + stripe_width))
                attacked[start:end, :] += intensity
            else:
                start = max(0, min(width - 1, start))
                end = max(start + 1, min(width, start + stripe_width))
                attacked[:, start:end] += intensity

        elif attack_type == "block_dropout":
            x = int(payload.get("x", width // 3))
            y = int(payload.get("y", height // 3))
            block_width = max(1, int(payload.get("width", 100)))
            block_height = max(1, int(payload.get("height", 100)))
            fill_value = float(payload.get("fill_value", 0.0))
            x0 = max(0, min(width, x))
            y0 = max(0, min(height, y))
            x1 = max(x0 + 1, min(width, x0 + block_width))
            y1 = max(y0 + 1, min(height, y0 + block_height))
            attacked[y0:y1, x0:x1] = fill_value

        elif attack_type == "salt_pepper":
            amount = float(payload.get("amount", 0.002))
            salt_value = float(
                payload.get(
                    "salt_value",
                    np.nanmax(clean[np.isfinite(clean)]) if np.isfinite(clean).any() else 255.0,
                )
            )
            pepper_value = float(
                payload.get(
                    "pepper_value",
                    np.nanmin(clean[np.isfinite(clean)]) if np.isfinite(clean).any() else 0.0,
                )
            )
            seed = int(payload.get("seed", 42))
            rng = np.random.default_rng(seed)
            total = height * width
            count = max(1, int(total * amount))
            ys = rng.integers(0, height, size=count)
            xs = rng.integers(0, width, size=count)
            half = count // 2
            attacked[ys[:half], xs[:half]] = salt_value
            attacked[ys[half:], xs[half:]] = pepper_value

        return attacked

    def _detect_anomalies(clean: ImageArray, attacked: ImageArray, threshold: float = 8.0) -> Tuple[ImageArray, Dict[str, object]]:
        residual = attacked - clean
        finite = residual[np.isfinite(residual)]
        if finite.size == 0:
            mask = np.zeros(clean.shape, dtype=bool)
            return mask, {
                "status": "clean",
                "severity": "none",
                "score": 0.0,
                "coverage_ratio": 0.0,
                "overlay": None,
                "message": "No anomaly detected.",
            }

        median_res = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median_res)))
        mad = mad if mad > 1e-6 else 1e-6
        robust_z = 0.6745 * (residual - median_res) / mad
        mask = np.abs(robust_z) > threshold

        coords = np.argwhere(mask)
        coverage_ratio = float(mask.mean())
        max_score = float(np.max(np.abs(robust_z[np.isfinite(robust_z)]))) if np.isfinite(robust_z).any() else 0.0

        overlay = None
        if coords.size > 0:
            y0 = int(coords[:, 0].min())
            y1 = int(coords[:, 0].max())
            x0 = int(coords[:, 1].min())
            x1 = int(coords[:, 1].max())
            cy = int(np.round(coords[:, 0].mean()))
            cx = int(np.round(coords[:, 1].mean()))
            overlay = {
                "bbox": {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1},
                "centroid": {"x": cx, "y": cy},
                "pixel_count": int(coords.shape[0]),
            }

        if coords.size == 0:
            severity = "none"
            status = "clean"
            message = "No anomaly detected."
        elif coverage_ratio < 0.001 and max_score < 25:
            severity = "low"
            status = "flagged"
            message = "Low-severity anomaly detected. Frame flagged for review."
        elif coverage_ratio < 0.01 and max_score < 60:
            severity = "medium"
            status = "quarantined"
            message = "Medium-severity anomaly detected. Frame quarantined from downstream processing."
        else:
            severity = "high"
            status = "quarantined"
            message = "High-severity anomaly detected. Frame quarantined and downstream promotion blocked."

        return mask, {
            "status": status,
            "severity": severity,
            "score": round(max_score, 2),
            "coverage_ratio": round(coverage_ratio, 6),
            "overlay": overlay,
            "message": message,
        }

    def _summarize_containment(slug: str, anomaly: Dict[str, object], attack_active: bool) -> Dict[str, object]:
        state = _get_containment_state(slug)
        contained = bool(state.get("contained", False))
        mode = state.get("mode")
        disconnected = bool(state.get("disconnected", False))
        display_mode = str(state.get("display_mode", "attacked"))
        blocked_attempts = int(state.get("blocked_attempts", 0) or 0)

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

        return {
            "contained": contained,
            "mode": mode,
            "disconnected": disconnected,
            "display_mode": display_mode,
            "blocked_attempts": blocked_attempts,
            "message": message,
            "allow_manual_contain": not contained,
            "allow_disconnect": contained or attack_active,
            "allow_resume": contained or disconnected,
        }

    def _prepare_raw_view(slug: str) -> Optional[Dict[str, object]]:
        payload = _load_science_image(slug)
        if payload is None:
            return None

        clean_arr, meta = payload
        attack = ACTIVE_ATTACKS.get(slug)
        attack_active = attack is not None
        state = _get_containment_state(slug)

        attacked_arr = _apply_attack(clean_arr, attack) if attack_active else clean_arr.copy()
        anomaly_mask, anomaly = _detect_anomalies(clean_arr, attacked_arr)

        attack_signature = _attack_signature(attack)
        if (
            attack_active
            and anomaly.get("severity") in {"medium", "high"}
            and not state.get("contained")
            and state.get("suppressed_attack_signature") != attack_signature
        ):
            state.update(
                {
                    "contained": True,
                    "mode": "automatic",
                    "display_mode": "attacked",
                    "message": "Automatically contained due to anomaly severity.",
                }
            )

        display_base = clean_arr if state.get("display_mode") == "clean" else attacked_arr
        display_mask_source = np.zeros(clean_arr.shape, dtype=bool) if state.get("display_mode") == "clean" else anomaly_mask
        display_anomaly = dict(anomaly)
        if state.get("display_mode") == "clean":
            display_anomaly = {
                **display_anomaly,
                "overlay": None,
                "message": "Trusted baseline restored for inspection after analyst action.",
            }

        crop_bounds = _compute_valid_crop_bounds(clean_arr)
        display_source = _crop_with_bounds(display_base, crop_bounds)
        display_mask = _crop_with_bounds(display_mask_source, crop_bounds)
        display_arr = _normalize_to_uint8(display_source)

        display_overlay = display_anomaly.get("overlay")
        if crop_bounds is not None and display_overlay is not None:
            r0, _r1, c0, _c1 = crop_bounds
            bbox = display_overlay.get("bbox")
            centroid = display_overlay.get("centroid")
            if bbox:
                bbox["x"] = int(bbox["x"] - c0)
                bbox["y"] = int(bbox["y"] - r0)
            if centroid:
                centroid["x"] = int(centroid["x"] - c0)
                centroid["y"] = int(centroid["y"] - r0)

        containment = _summarize_containment(slug, anomaly, attack_active)
        cropped_meta = {
            **meta,
            "display_shape": list(display_arr.shape),
            "crop_bounds": {
                "row_start": crop_bounds[0],
                "row_end": crop_bounds[1],
                "col_start": crop_bounds[2],
                "col_end": crop_bounds[3],
            } if crop_bounds is not None else None,
        }

        return {
            "clean": clean_arr,
            "attacked": attacked_arr,
            "display": display_arr,
            "display_mask": display_mask,
            "anomaly_mask": anomaly_mask,
            "anomaly": display_anomaly,
            "raw_anomaly": anomaly,
            "attack_active": attack_active,
            "attack": attack,
            "containment": containment,
            "meta": cropped_meta,
        }

    def _compute_histogram(values: ImageArray, bins: int = 20, value_range: Tuple[float, float] = (0, 255)) -> List[int]:
        hist, _ = np.histogram(values, bins=bins, range=value_range)
        return hist.astype(int).tolist()

    @app.route("/")
    def home():
        images: List[Dict[str, str]] = []
        if rendered_base.exists():
            for img in sorted(rendered_base.iterdir()):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    images.append(
                        {
                            "filename": img.name,
                            "label": img.stem.replace("_", " ").title(),
                            "url": url_for("serve_rendered", filename=img.name),
                        }
                    )
        return render_template("index.html", images=images, current_year=2026, nav_items=build_nav())

    @app.route("/rendered/<path:filename>")
    def serve_rendered(filename: str):
        return send_from_directory(rendered_base, filename)

    @app.route("/raw/<slug>")
    def raw_page(slug: str):
        entry = RAW_OBJECTS.get(slug)
        if entry is None:
            return render_template("error.html", message="Unknown raw dataset"), 404
        title, _path = entry
        return render_template(
            "raw.html",
            title=title,
            slug=slug,
            nav_items=build_nav(),
            current_year=2026,
        )

    @app.route("/api/raw_data/<slug>")
    def api_raw_data(slug: str):
        view = _prepare_raw_view(slug)
        if view is None:
            return jsonify({"error": "Failed to parse FITS or no usable 2-D image found."}), 500

        arr = view["display"]
        return jsonify(
            {
                "width": int(arr.shape[1]),
                "height": int(arr.shape[0]),
                "data": arr.tolist(),
                "meta": view["meta"],
                "attack_active": view["attack_active"],
                "attack": view["attack"],
                "anomaly": view["anomaly"],
                "containment": view["containment"],
            }
        )

    @app.route("/api/raw_stats/<slug>")
    def api_raw_stats(slug: str):
        view = _prepare_raw_view(slug)
        if view is None:
            return jsonify({"error": "Failed to parse FITS or no usable 2-D image found."}), 500

        display = np.asarray(view["display"], dtype=np.uint8)
        display_mask = np.asarray(view["display_mask"], dtype=bool)
        flat = display.astype(np.float32).ravel()
        anomaly_flat = display[display_mask].astype(np.float32) if display_mask.any() else np.array([], dtype=np.float32)

        stats = {
            "min": float(flat.min()) if flat.size else None,
            "max": float(flat.max()) if flat.size else None,
            "mean": float(flat.mean()) if flat.size else None,
            "median": float(np.median(flat)) if flat.size else None,
            "std": float(flat.std()) if flat.size else None,
            "histogram": _compute_histogram(flat),
            "anomaly_histogram": _compute_histogram(anomaly_flat) if anomaly_flat.size else [0] * 20,
            "attack_active": view["attack_active"],
            "attack": view["attack"],
            "anomaly": view["anomaly"],
            "containment": view["containment"],
            "meta": view["meta"],
        }
        return jsonify(stats)

    @app.route("/api/attack_status/<slug>")
    def api_attack_status(slug: str):
        if slug not in RAW_OBJECTS:
            return jsonify({"error": "Unknown raw dataset"}), 404
        view = _prepare_raw_view(slug)
        if view is None:
            return jsonify({"error": "Failed to load dataset"}), 500
        return jsonify(
            {
                "slug": slug,
                "attack_active": view["attack_active"],
                "attack": view["attack"],
                "anomaly": view["anomaly"],
                "containment": view["containment"],
                "meta": view["meta"],
            }
        )

    @app.route("/api/session_start", methods=["POST"])
    def api_session_start():
        payload = request.get_json(silent=True) or {}
        slug = str(payload.get("slug", "")).strip().lower()
        client_name = str(payload.get("client_name", "raspberry-pi")).strip() or "raspberry-pi"

        if not slug:
            return jsonify({"error": "Missing 'slug'."}), 400
        if slug not in RAW_OBJECTS:
            return jsonify({"error": "Unknown slug."}), 404

        state = _get_containment_state(slug)
        if state.get("disconnected"):
            return jsonify(
                {
                    "error": "Attack source currently disconnected for this dataset.",
                    "slug": slug,
                    "containment": _summarize_containment(slug, {"severity": "none"}, slug in ACTIVE_ATTACKS),
                }
            ), 423

        session = _start_attack_session(slug, client_name)
        return jsonify(
            {
                "message": "Attack session started.",
                "session_id": session["session_id"],
                "slug": slug,
                "client_name": client_name,
            }
        )

    @app.route("/api/session_status/<session_id>")
    def api_session_status(session_id: str):
        session = _get_attack_session(session_id.strip())
        if session is None:
            return jsonify({"error": "Unknown session_id."}), 404

        return jsonify(
            {
                "session_id": session["session_id"],
                "slug": session["slug"],
                "client_name": session["client_name"],
                "active": bool(session.get("active", False)),
                "disconnect_reason": session.get("disconnect_reason"),
            }
        )

    @app.route("/api/attack_ingest", methods=["POST"])
    def api_attack_ingest():
        payload = request.get_json(silent=True) or {}
        ok, message = _validate_attack_payload(payload)
        if not ok:
            return jsonify({"error": message}), 400

        slug = str(payload.get("slug", "")).strip().lower()
        session_id = str(payload.get("session_id", "")).strip() or None
        state = _get_containment_state(slug)

        if session_id:
            session = _get_attack_session(session_id)
            if session is None:
                return jsonify({"error": "Unknown session_id."}), 404
            if not session.get("active", False):
                return jsonify(
                    {
                        "message": "Attack session inactive. Payload rejected.",
                        "slug": slug,
                        "session_id": session_id,
                        "blocked": True,
                        "disconnect_reason": session.get("disconnect_reason"),
                    }
                ), 423

        if state.get("disconnected"):
            state["blocked_attempts"] = int(state.get("blocked_attempts", 0) or 0) + 1
            if session_id and _get_attack_session(session_id) is not None:
                _get_attack_session(session_id)["active"] = False
                _get_attack_session(session_id)["disconnect_reason"] = "disconnected_by_defender"
            return jsonify(
                {
                    "message": "Attack source disconnected. Incoming attack blocked.",
                    "slug": slug,
                    "session_id": session_id,
                    "blocked": True,
                    "containment": _summarize_containment(slug, {"severity": "none"}, slug in ACTIVE_ATTACKS),
                }
            ), 423

        ACTIVE_ATTACKS[slug] = {
            **payload,
            "slug": slug,
            "attack_type": str(payload.get("attack_type", "")).strip().lower(),
        }
        state["suppressed_attack_signature"] = None

        view = _prepare_raw_view(slug)
        return jsonify(
            {
                "message": "Attack accepted and applied to the raw visualization layer.",
                "slug": slug,
                "session_id": session_id,
                "attack": ACTIVE_ATTACKS[slug],
                "anomaly": view["anomaly"] if view else None,
                "containment": view["containment"] if view else None,
            }
        )

    @app.route("/api/contain/<slug>", methods=["POST"])
    def api_contain(slug: str):
        slug = slug.strip().lower()
        if slug not in RAW_OBJECTS:
            return jsonify({"error": "Unknown raw dataset"}), 404

        state = _get_containment_state(slug)
        state.update(
            {
                "contained": True,
                "mode": "manual",
                "display_mode": "attacked",
                "message": "Manual containment enabled for analyst inspection.",
            }
        )
        view = _prepare_raw_view(slug)
        return jsonify(
            {
                "message": "Manual containment enabled.",
                "slug": slug,
                "containment": view["containment"] if view else state,
            }
        )

    @app.route("/api/disconnect_attacker/<slug>", methods=["POST"])
    def api_disconnect_attacker(slug: str):
        slug = slug.strip().lower()
        if slug not in RAW_OBJECTS:
            return jsonify({"error": "Unknown raw dataset"}), 404

        ACTIVE_ATTACKS.pop(slug, None)
        disconnected_sessions = _disconnect_sessions_for_slug(slug)
        state = _get_containment_state(slug)
        state.update(
            {
                "contained": True,
                "mode": state.get("mode") or "manual",
                "display_mode": "clean",
                "disconnected": True,
                "message": "Suspected attack source disconnected. Trusted baseline restored.",
            }
        )
        view = _prepare_raw_view(slug)
        return jsonify(
            {
                "message": "Attacker disconnected and trusted baseline restored.",
                "slug": slug,
                "terminated_sessions": disconnected_sessions,
                "containment": view["containment"] if view else state,
            }
        )

    @app.route("/api/release/<slug>", methods=["POST"])
    def api_release(slug: str):
        slug = slug.strip().lower()
        if slug not in RAW_OBJECTS:
            return jsonify({"error": "Unknown raw dataset"}), 404

        state = _reset_containment_state(slug)
        if slug in ACTIVE_ATTACKS:
            state["suppressed_attack_signature"] = _attack_signature(ACTIVE_ATTACKS.get(slug))
        view = _prepare_raw_view(slug)
        return jsonify(
            {
                "message": "Containment released. Monitoring resumed.",
                "slug": slug,
                "containment": view["containment"] if view else state,
            }
        )

    @app.route("/api/attack_clear/<slug>", methods=["POST"])
    def api_attack_clear(slug: str):
        slug = slug.strip().lower()
        removed = ACTIVE_ATTACKS.pop(slug, None)
        state = _get_containment_state(slug)
        if not state.get("disconnected"):
            state["display_mode"] = "attacked"
        state["suppressed_attack_signature"] = None
        return jsonify({"message": "Attack state cleared.", "slug": slug, "cleared": removed is not None})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
