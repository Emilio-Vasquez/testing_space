"""
Flask application for the Astrophysics Cybersecurity dashboard.

This version fixes raw JWST/HST FITS display by using Astropy instead of a
manual FITS parser.  In particular, many JWST products store the displayable
image in an extension such as ``SCI`` rather than in the primary HDU.

Key improvements
----------------
- Uses ``astropy.io.fits`` to read FITS reliably.
- Prefers a 2-D ``SCI`` extension when present.
- Falls back to the first usable 2-D image HDU.
- Handles NaNs/infinities and robust percentile normalization.
- Caches normalized image data so repeated API calls are faster.
- Recursively discovers FITS files under ``data/fits/mastDownload``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from flask import Flask, jsonify, render_template, send_from_directory, url_for


ImageArray = np.ndarray


def create_app() -> Flask:
    """Create and configure the Flask application."""
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
        raise RuntimeError(
            "Cannot locate data directory. Ensure data/fits/mastDownload exists."
        )

    fits_base = data_base / "fits" / "mastDownload"
    rendered_base = data_base / "rendered"

    SLUG_TITLE_OVERRIDES: Dict[str, str] = {
        "ice00ap1q_raw": "Pillars of Creation (M16)",
        "hst_8063_p7_nic_nic3_f110w_01_drz": "Stephan’s Quintet",
        "jw02736006001_02101_00001_nrs1_cal": "SMACS 0723 Deep Field (JWST)",
    }

    def discover_raw_files() -> Dict[str, Tuple[str, Path]]:
        """Recursively scan for FITS files and map slug -> (title, path)."""
        mapping: Dict[str, Tuple[str, Path]] = {}
        if not fits_base.exists():
            return mapping

        patterns = ("*.fits", "*.fit", "*.fts")
        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(fits_base.rglob(pattern))

        for file in sorted(paths):
            if not file.is_file():
                continue
            slug = file.stem.replace(" ", "_").lower()
            title = SLUG_TITLE_OVERRIDES.get(slug, slug.replace("_", " ").title())
            mapping[slug] = (title, file)
        return mapping

    RAW_OBJECTS: Dict[str, Tuple[str, Path]] = discover_raw_files()

    def _find_best_image_hdu(hdul: fits.HDUList) -> Optional[Tuple[int, str, ImageArray]]:
        """Return the best 2-D image HDU as (index, name, data).

        Preference order:
        1. Image extension named SCI with 2-D data.
        2. Any 2-D image extension.
        3. A higher-dimensional image collapsed to its first 2-D plane.
        """

        def normalize_hdu_data(data: object) -> Optional[ImageArray]:
            if data is None:
                return None
            arr = np.asarray(data)
            if arr.size == 0:
                return None
            if arr.ndim == 2:
                return arr
            if arr.ndim > 2:
                # Collapse by repeatedly selecting the first plane until 2-D.
                while arr.ndim > 2:
                    arr = arr[0]
                if arr.ndim == 2:
                    return arr
            return None

        # Prefer SCI first.
        for idx, hdu in enumerate(hdul):
            extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            arr = normalize_hdu_data(getattr(hdu, "data", None))
            if extname == "SCI" and arr is not None:
                return idx, extname or f"HDU{idx}", arr

        # Then any usable image HDU.
        for idx, hdu in enumerate(hdul):
            extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            arr = normalize_hdu_data(getattr(hdu, "data", None))
            if arr is not None:
                return idx, extname or f"HDU{idx}", arr

        return None

    def _robust_normalize_to_uint8(arr: ImageArray) -> Optional[ImageArray]:
        """Convert science data to an 8-bit grayscale image for browser display."""
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None

        # Robust clipping for astronomical images.
        vmin, vmax = np.percentile(finite, [0.5, 99.5])
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if vmax <= vmin:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if vmax <= vmin:
                return np.zeros(arr.shape, dtype=np.uint8)

        clipped = np.clip(arr, vmin, vmax)
        scaled = (clipped - vmin) / (vmax - vmin)
        scaled = np.clip(scaled * 255.0, 0, 255)
        return scaled.astype(np.uint8)

    @lru_cache(maxsize=32)
    def _load_display_image(slug: str) -> Optional[Tuple[List[List[int]], Dict[str, object]]]:
        """Load, select, and normalize the image used by the API endpoints."""
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
                display_arr = _robust_normalize_to_uint8(raw_arr)
                if display_arr is None:
                    return None

                metadata = {
                    "source_file": path.name,
                    "selected_hdu_index": hdu_index,
                    "selected_hdu_name": hdu_name,
                    "shape": list(display_arr.shape),
                    "dtype": str(raw_arr.dtype),
                }
                return display_arr.tolist(), metadata
        except Exception as exc:
            print(f"[ERROR] Failed to read FITS for slug '{slug}': {exc}")
            return None

    def compute_stats(arr: List[List[int]]) -> Dict[str, object]:
        flat = np.array(arr, dtype=np.float32).ravel()
        hist, _ = np.histogram(flat, bins=20, range=(0, 255))
        return {
            "min": float(flat.min()) if flat.size else None,
            "max": float(flat.max()) if flat.size else None,
            "mean": float(flat.mean()) if flat.size else None,
            "median": float(np.median(flat)) if flat.size else None,
            "std": float(flat.std()) if flat.size else None,
            "histogram": hist.tolist(),
        }

    def build_nav() -> List[Dict[str, str]]:
        preferred_order = [
            "ice00ap1q_raw",                          # Pillars of Creation? wait user wants Home, Stephan's, JWST, Pillars
            "hst_8063_p7_nic_nic3_f110w_01_drz",
            "jw02736006001_02101_00001_nrs1_cal",
        ]
        
        # user requested: Home, Stephan's, JWST, then Pillars
        preferred_order = [
            "hst_8063_p7_nic_nic3_f110w_01_drz",      # Stephan’s Quintet
            "jw02736006001_02101_00001_nrs1_cal",    # JWST
            "ice00ap1q_raw",                         # Pillars of Creation
        ]

        nav_items = []

        for slug in preferred_order:
            if slug in RAW_OBJECTS:
                title, _path = RAW_OBJECTS[slug]
                nav_items.append({"slug": slug, "title": title})

        # add any other discovered files afterward
        for slug, (title, _path) in RAW_OBJECTS.items():
            if slug not in preferred_order:
                nav_items.append({"slug": slug, "title": title})

        return nav_items

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
        return render_template(
            "index.html",
            images=images,
            current_year=2026,
            nav_items=build_nav(),
        )

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
        payload = _load_display_image(slug)
        if payload is None:
            return jsonify({"error": "Failed to parse FITS or no usable 2-D image found."}), 500

        arr, metadata = payload
        if not arr or not arr[0]:
            return jsonify({"error": "Parsed FITS image is empty."}), 500

        return jsonify(
            {
                "width": len(arr[0]),
                "height": len(arr),
                "data": arr,
                "meta": metadata,
            }
        )

    @app.route("/api/raw_stats/<slug>")
    def api_raw_stats(slug: str):
        payload = _load_display_image(slug)
        if payload is None:
            return jsonify({"error": "Failed to parse FITS or no usable 2-D image found."}), 500

        arr, metadata = payload
        stats = compute_stats(arr)
        stats["meta"] = metadata
        return jsonify(stats)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
