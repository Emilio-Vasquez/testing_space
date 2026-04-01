"""
Flask application for the Astrophysics Cybersecurity dashboard.

This backend serves two main purposes:

1. Display static “science product” images on the home page.  These
   images are the final calibrated outputs from real missions and are
   stored in ``data/rendered``.
2. Stream raw data from FITS exposures line by line, compute summary
   statistics and a histogram of pixel intensities, and provide
   endpoints for the front‑end to fetch these data.  The raw exposures
   are real science frames (usually calibration level 2) downloaded
   from MAST and saved in ``data/fits/mastDownload``.

The application automatically discovers available FITS files and
renders navigation links for each.  When a new FITS is downloaded (by
running ``data_fetcher.py``), it will appear in the navigation the
next time the server is started.

Anomaly detection is not performed here; it should be implemented in
separate pipeline scripts and the results can be surfaced via JSON.

To run the application:

    python app.py

Ensure you have installed Flask via ``pip install flask``.  For
production use behind a reverse proxy, configure a production‑grade
WSGI server such as gunicorn.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, send_from_directory, url_for


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Determine where to look for data.  We support data stored under
    # ``dashboard/data`` (relative to this file), ``ADPs/data`` (two
    # levels up when running from within the ADPs package), and a
    # project root ``data`` directory.  The first that contains
    # ``fits/mastDownload`` is used.
    current_dir = Path(__file__).resolve().parent
    base_candidates = [
        current_dir / "data",
        current_dir.parents[1] / "data",
        current_dir.parents[2] / "data",
    ]
    data_base: Optional[Path] = None
    for candidate in base_candidates:
        if (candidate / "fits" / "mastDownload").exists():
            data_base = candidate
            break
    if data_base is None:
        raise RuntimeError(
            "Cannot locate data directory. Ensure data/fits and data/rendered exist."
        )

    fits_base = data_base / "fits" / "mastDownload"
    rendered_base = data_base / "rendered"

    # Override the human‑readable titles for certain slugs.  This allows
    # nicer names in the navigation bar instead of raw filenames.
    SLUG_TITLE_OVERRIDES: Dict[str, str] = {
        # Extend this dictionary with your own mappings.  The keys are
        # the slugified filenames (lowercase, spaces replaced by
        # underscores, no extension) and the values are the labels
        # shown in the navigation bar.
        "ice00ap1q_raw": "Pillars of Creation (M16)",
        "hst_8063_p7_nic_nic3_f110w_01_drz": "Stephan’s Quintet",
        "jw02736006001_02101_00001_nrs1_cal": "SMACS 0723 Deep Field (JWST)",
    }

    def discover_raw_files() -> Dict[str, Tuple[str, Path]]:
        """Scan the fits_base directory for FITS files and build a slug→(title, path) map.

        The slug is derived from the filename and used as the URL key.
        A human‑friendly title is computed from the slug but can be
        overridden via SLUG_TITLE_OVERRIDES.
        """
        mapping: Dict[str, Tuple[str, Path]] = {}
        if not fits_base.exists():
            return mapping
        for mission_dir in fits_base.iterdir():
            if not mission_dir.is_dir():
                continue
            for obs_dir in mission_dir.iterdir():
                if not obs_dir.is_dir():
                    continue
                for file in obs_dir.iterdir():
                    if file.suffix.lower().endswith(".fits"):
                        slug = file.stem.replace(" ", "_").lower()
                        title = slug.replace("_", " ").title()
                        if slug in SLUG_TITLE_OVERRIDES:
                            title = SLUG_TITLE_OVERRIDES[slug]
                        mapping[slug] = (title, file)
        return mapping

    RAW_OBJECTS: Dict[str, Tuple[str, Path]] = discover_raw_files()

    def _parse_fits_image(path: Path) -> Optional[List[List[int]]]:
        """Parse a FITS file and return a 2‑D array of pixel intensities.

        The parser reads the primary HDU header to determine whether
        image data exist in the primary or extension HDUs.  It supports
        integer (BITPIX > 0) and floating point (BITPIX < 0) pixel types
        and will collapse higher dimensional data cubes by taking the
        first plane.  For JWST and HST files, the parser prefers an
        extension named ``SCI`` if one exists.  If the file cannot be
        parsed, ``None`` is returned.
        """
        try:
            with path.open("rb") as fh:
                # Read the primary header blocks
                hdr_blocks: List[str] = []
                while True:
                    blk = fh.read(2880).decode("ascii", "ignore")
                    hdr_blocks.append(blk)
                    if "END" in blk:
                        break
                prim_hdr = "".join(hdr_blocks)
                prim_cards: Dict[str, str] = {}
                for i in range(0, len(prim_hdr), 80):
                    line = prim_hdr[i : i + 80]
                    key = line[:8].strip()
                    if "=" in line:
                        val = line[10:].split("/")[0].strip()
                    else:
                        val = None
                    if key:
                        prim_cards[key] = val
                # Establish offset after primary header
                data_offset = len(hdr_blocks) * 2880
                # Gather extension headers and offsets
                nextend = int(prim_cards.get("NEXTEND", "0") or 0)
                ext_headers: List[Tuple[Dict[str, str], int]] = []
                for _ in range(nextend):
                    ext_blocks: List[str] = []
                    while True:
                        blk = fh.read(2880).decode("ascii", "ignore")
                        ext_blocks.append(blk)
                        if "END" in blk:
                            break
                    ext_hdr = "".join(ext_blocks)
                    ext_cards: Dict[str, str] = {}
                    for j in range(0, len(ext_hdr), 80):
                        ln = ext_hdr[j : j + 80]
                        k = ln[:8].strip()
                        if "=" in ln:
                            v = ln[10:].split("/")[0].strip()
                        else:
                            v = None
                        if k:
                            ext_cards[k] = v
                    ext_offset = data_offset
                    data_offset += len(ext_blocks) * 2880
                    ext_headers.append((ext_cards, ext_offset))
                # Determine which header to use (SCI preferred)
                used_cards: Optional[Dict[str, str]] = None
                used_offset: Optional[int] = None
                # First look for a SCI extension with image keywords
                for cards, offset in ext_headers:
                    extname = cards.get("EXTNAME", "")
                    extname_str = extname.strip("'\" ").upper() if extname else ""
                    if (
                        extname_str == "SCI"
                        and all(k in cards for k in ("NAXIS", "NAXIS1", "NAXIS2", "BITPIX"))
                    ):
                        used_cards = cards
                        used_offset = offset
                        break
                # If no SCI, fall back to first image extension
                if used_cards is None:
                    for cards, offset in ext_headers:
                        if all(k in cards for k in ("NAXIS", "NAXIS1", "NAXIS2", "BITPIX")):
                            used_cards = cards
                            used_offset = offset
                            break
                # If still none, try primary header
                if used_cards is None and all(k in prim_cards for k in ("NAXIS", "NAXIS1", "NAXIS2", "BITPIX")):
                    used_cards = prim_cards
                    used_offset = len(hdr_blocks) * 2880
                if used_cards is None or used_offset is None:
                    return None
                # Extract image dimensions and bit depth
                naxis = int(used_cards["NAXIS"])
                width = int(used_cards["NAXIS1"])
                height = int(used_cards["NAXIS2"])
                bitpix = int(used_cards["BITPIX"])
                # Collapse higher dimensions by choosing the first plane
                if naxis >= 3 and "NAXIS3" in used_cards:
                    naxis3 = int(used_cards["NAXIS3"])
                # Seek to the data for the selected extension
                fh.seek(used_offset)
                bytes_per = abs(bitpix) // 8
                # Determine numpy dtype
                if bitpix > 0:
                    if bitpix == 8:
                        dt = ">i1"
                    elif bitpix == 16:
                        dt = ">i2"
                    elif bitpix == 32:
                        dt = ">i4"
                    else:
                        return None
                else:
                    if bitpix == -32:
                        dt = ">f4"
                    elif bitpix == -64:
                        dt = ">f8"
                    else:
                        return None
                slice_bytes = width * height * bytes_per
                raw_data = fh.read(slice_bytes)
                import numpy as np  # dynamic import
                arr = np.frombuffer(raw_data, dtype=dt)
                if arr.size != width * height:
                    return None
                arr = arr.reshape((height, width)).astype(float)
                # Drop NaNs and infinities
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return None
                # Use percentile clipping to enhance contrast
                vmin, vmax = np.percentile(finite, [0.5, 99.5])
                arr = np.clip(arr, vmin, vmax)
                if np.max(arr) == np.min(arr):
                    return None
                norm = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
                return norm.astype(int).tolist()
        except Exception:
            return None

    def compute_stats(arr: List[List[int]]) -> Dict[str, object]:
        import numpy as np
        flat = np.array(arr).astype(float).ravel()
        hist, _ = np.histogram(flat, bins=20)
        return {
            "min": float(flat.min()) if flat.size else None,
            "max": float(flat.max()) if flat.size else None,
            "mean": float(flat.mean()) if flat.size else None,
            "median": float(np.median(flat)) if flat.size else None,
            "std": float(flat.std()) if flat.size else None,
            "histogram": hist.tolist(),
        }

    def build_nav() -> List[Dict[str, str]]:
        return [
            {"slug": slug, "title": title}
            for slug, (title, _path) in RAW_OBJECTS.items()
        ]

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
        entry = RAW_OBJECTS.get(slug)
        if entry is None:
            return jsonify({"error": "Unknown raw dataset"}), 404
        _title, path = entry
        arr = _parse_fits_image(path)
        if arr is None:
            return jsonify({"error": "Failed to parse FITS"}), 500
        return jsonify({"width": len(arr[0]), "height": len(arr), "data": arr})

    @app.route("/api/raw_stats/<slug>")
    def api_raw_stats(slug: str):
        entry = RAW_OBJECTS.get(slug)
        if entry is None:
            return jsonify({"error": "Unknown raw dataset"}), 404
        _title, path = entry
        arr = _parse_fits_image(path)
        if arr is None:
            return jsonify({"error": "Failed to parse FITS"}), 500
        stats = compute_stats(arr)
        return jsonify(stats)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)