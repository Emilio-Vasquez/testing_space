from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

ImageArray = np.ndarray


def detect_anomalies(clean: ImageArray, attacked: ImageArray, threshold: float = 8.0) -> Tuple[ImageArray, Dict[str, object]]:
    residual = attacked - clean
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        mask = np.zeros(clean.shape, dtype=bool)
        return mask, {"status": "clean", "severity": "none", "score": 0.0, "coverage_ratio": 0.0, "overlay": None, "message": "No anomaly detected."}
    median_res = float(np.median(finite)); mad = float(np.median(np.abs(finite - median_res))); mad = mad if mad > 1e-6 else 1e-6
    robust_z = 0.6745 * (residual - median_res) / mad; mask = np.abs(robust_z) > threshold
    coords = np.argwhere(mask); coverage_ratio = float(mask.mean())
    max_score = float(np.max(np.abs(robust_z[np.isfinite(robust_z)]))) if np.isfinite(robust_z).any() else 0.0
    overlay = None
    if coords.size > 0:
        y0 = int(coords[:, 0].min()); y1 = int(coords[:, 0].max()); x0 = int(coords[:, 1].min()); x1 = int(coords[:, 1].max())
        cy = int(np.round(coords[:, 0].mean())); cx = int(np.round(coords[:, 1].mean()))
        overlay = {"bbox": {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1}, "centroid": {"x": cx, "y": cy}, "pixel_count": int(coords.shape[0])}
    if coords.size == 0:
        severity = "none"; status = "clean"; message = "No anomaly detected."
    elif coverage_ratio < 0.001 and max_score < 25:
        severity = "low"; status = "flagged"; message = "Low-severity anomaly detected. Frame flagged for review."
    elif coverage_ratio < 0.01 and max_score < 60:
        severity = "medium"; status = "quarantined"; message = "Medium-severity anomaly detected. Frame quarantined from downstream processing."
    else:
        severity = "high"; status = "quarantined"; message = "High-severity anomaly detected. Frame quarantined and downstream promotion blocked."
    return mask, {"status": status, "severity": severity, "score": round(max_score, 2), "coverage_ratio": round(coverage_ratio, 6), "overlay": overlay, "message": message}
