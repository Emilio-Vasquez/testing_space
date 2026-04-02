from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .catalog import get_raw_objects
from .sessions import get_attack_session

ImageArray = np.ndarray


def validate_attack_payload(payload: Dict[str, object]) -> Tuple[bool, str]:
    slug = str(payload.get("slug", "")).strip().lower()
    if not slug:
        return False, "Missing 'slug'."
    if slug not in get_raw_objects():
        return False, "Unknown slug."
    attack_type = str(payload.get("attack_type", "")).strip().lower()
    allowed = {"hotspot", "stripe_noise", "block_dropout", "salt_pepper"}
    if attack_type not in allowed:
        return False, f"Unsupported attack_type. Allowed values: {sorted(allowed)}"
    session_id = str(payload.get("session_id", "")).strip()
    if session_id:
        session = get_attack_session(session_id)
        if session is None:
            return False, "Unknown session_id."
        if str(session.get("slug", "")).strip().lower() != slug:
            return False, "session_id does not match slug."
        if not session.get("active", False):
            return False, "Session is no longer active."
    return True, "ok"


def apply_attack(clean: ImageArray, payload: Dict[str, object]) -> ImageArray:
    attacked = np.array(clean, copy=True, dtype=np.float32)
    attack_type = str(payload.get("attack_type", "")).lower()
    height, width = attacked.shape
    if attack_type == "hotspot":
        x = int(payload.get("x", width // 2)); y = int(payload.get("y", height // 2))
        radius = max(1, int(payload.get("radius", 20))); intensity = float(payload.get("intensity", 4000.0))
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        attacked[mask] += intensity
    elif attack_type == "stripe_noise":
        axis = str(payload.get("axis", "vertical")).lower()
        start = int(payload.get("start", width // 2 if axis == "vertical" else height // 2))
        stripe_width = max(1, int(payload.get("width", 15))); intensity = float(payload.get("intensity", 1500.0))
        if axis == "horizontal":
            start = max(0, min(height - 1, start)); end = max(start + 1, min(height, start + stripe_width))
            attacked[start:end, :] += intensity
        else:
            start = max(0, min(width - 1, start)); end = max(start + 1, min(width, start + stripe_width))
            attacked[:, start:end] += intensity
    elif attack_type == "block_dropout":
        x = int(payload.get("x", width // 3)); y = int(payload.get("y", height // 3))
        block_width = max(1, int(payload.get("width", 100))); block_height = max(1, int(payload.get("height", 100)))
        fill_value = float(payload.get("fill_value", 0.0))
        x0 = max(0, min(width, x)); y0 = max(0, min(height, y))
        x1 = max(x0 + 1, min(width, x0 + block_width)); y1 = max(y0 + 1, min(height, y0 + block_height))
        attacked[y0:y1, x0:x1] = fill_value
    elif attack_type == "salt_pepper":
        amount = float(payload.get("amount", 0.002))
        salt_value = float(payload.get("salt_value", np.nanmax(clean[np.isfinite(clean)]) if np.isfinite(clean).any() else 255.0))
        pepper_value = float(payload.get("pepper_value", np.nanmin(clean[np.isfinite(clean)]) if np.isfinite(clean).any() else 0.0))
        seed = int(payload.get("seed", 42)); rng = np.random.default_rng(seed)
        total = height * width; count = max(1, int(total * amount))
        ys = rng.integers(0, height, size=count); xs = rng.integers(0, width, size=count)
        half = count // 2; attacked[ys[:half], xs[:half]] = salt_value; attacked[ys[half:], xs[half:]] = pepper_value
    return attacked
