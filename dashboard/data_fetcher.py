"""
data_fetcher.py – MAST Data Acquisition
=====================================

This script retrieves scientific exposures and preview images from the
Mikulski Archive for Space Telescopes (MAST) for a set of targets.  It
implements the following behaviour:

* For each target, search the archive for exposures in a given mission
  (`obs_collection`).  Because truly raw (calib_level 0) data are rarely
  released publicly, the script accepts a list of calibration levels
  ordered from most to least desirable (e.g. `[2, 1, 0]`).  The
  first available level is treated as the “raw” input for the
  pipeline.
* Download up to ``max_files`` exposures at that level.  Exposures
  are saved into ``data/fits/mastDownload/<mission>/<obs_id>/<filename>``.
* Fetch the science‑product preview (calib_level 3) images using the
  CAOM filtered service and save the first preview for each target
  into ``data/rendered``.  If no preview is available the script
  simply logs the fact.
* Keep track of previously downloaded observation IDs in a JSON file
  under ``data/processed/observations.json``.  On subsequent runs the
  script will only download new data.

The script can be invoked manually (``python data_fetcher.py``) or
scheduled to run periodically (e.g. via cron or APScheduler) to
incrementally collect new data as they become public in the archive.

The downloaded FITS files and preview images are the only data
sources used by the application; no synthetic samples are created.

Dependencies: ``astroquery`` (for observation queries) and
``requests`` (for image downloads).  Install them with

    pip install astroquery requests

Network access is required to talk to the MAST services.

"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import requests

try:
    from astroquery.mast import Observations
except ImportError as exc:
    raise SystemExit(
        "astroquery is required. Please install it with `pip install astroquery`."
    ) from exc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directories relative to this file.  These will be created if
# they do not exist.  If you prefer to store data elsewhere, modify
# these paths accordingly.
ROOT = Path(__file__).resolve().parents[1]  # two levels up
DATA_DIR = ROOT / "data"
FITS_DIR = DATA_DIR / "fits" / "mastDownload"
RENDERED_DIR = DATA_DIR / "rendered"
PROCESSED_FILE = DATA_DIR / "processed" / "observations.json"


def ensure_dirs() -> None:
    """Ensure all required directories exist."""
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_processed() -> Dict[str, List[str]]:
    """Load processed observation IDs from JSON.  Returns empty dict if file missing."""
    if PROCESSED_FILE.exists():
        try:
            return json.loads(PROCESSED_FILE.read_text())
        except Exception:
            logger.warning("Failed to read processed file; starting fresh.")
    return {}


def save_processed(proc: Dict[str, List[str]]) -> None:
    """Persist processed observation IDs to JSON."""
    PROCESSED_FILE.write_text(json.dumps(proc, indent=2))


def fetch_preview_urls(target_name: str, obs_collection: str, calib_level: int = 3) -> List[str]:
    """Return a list of JPEG preview URLs for the given target and mission.

    The CAOM filtered service is used via HTTP GET.  If no data are
    returned, an empty list is returned.
    """
    service_url = "https://mast.stsci.edu/api/v0/invoke"
    urls: List[str] = []
    # Request only the first page to avoid pulling hundreds of previews
    payload = {
        "service": "Mast.Caom.Filtered",
        "params": {
            "columns": "obs_id,jpegURL",
            "filters": [
                {"paramName": "obs_collection", "values": [obs_collection]},
                {"paramName": "target_name", "values": [target_name]},
                {"paramName": "calib_level", "values": [calib_level]},
            ],
        },
        "format": "json",
        "pagesize": 10,
        "page": 1,
    }
    try:
        response = requests.get(service_url, params={"request": json.dumps(payload)}, timeout=60)
        response.raise_for_status()
        rows = response.json().get("data", [])
        for row in rows:
            url = row.get("jpegURL")
            if url:
                urls.append(url)
    except Exception as exc:
        logger.error("Preview fetch failed for %s: %s", target_name, exc)
    return urls


def download_file(url: str, dest: Path) -> None:
    """Download a file from ``url`` to ``dest``, creating parent directories."""
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
    except Exception as exc:
        logger.error("Download failed for %s: %s", url, exc)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            fh.write(chunk)
    logger.info("Downloaded %s", dest)


def download_science_fits(
    target_name: str,
    obs_collection: str,
    levels: List[int],
    processed: Dict[str, List[str]],
    max_files: int = 2,
) -> None:
    """Download up to ``max_files`` science exposures for ``target_name``.

    This function iterates through the ``levels`` list, starting with the
    highest priority.  For the first level with available exposures, it
    downloads science products (productType="SCIENCE") until ``max_files``
    files have been saved.  Already processed observation IDs are
    skipped.  New obs_ids are recorded in ``processed``.
    """
    try:
        # Query MAST around the target with a generous radius (0.2 deg)
        table = Observations.query_object(target_name, radius="0.2 deg")
    except Exception as exc:
        logger.error("Observation query failed for %s: %s", target_name, exc)
        return
    table = table[table["obs_collection"] == obs_collection]
    if len(table) == 0:
        logger.info("No observations for %s in %s", target_name, obs_collection)
        return
    processed_ids = processed.get(target_name, [])
    for level in levels:
        subset = table[table["calib_level"] == level]
        if len(subset) == 0:
            continue
        new = subset[~np.isin(subset["obs_id"], processed_ids)]
        if len(new) == 0:
            continue
        logger.info("Found %d new obs for %s at level %d", len(new), target_name, level)
        products = Observations.get_product_list(new)
        sci = Observations.filter_products(products, productType="SCIENCE", mrp_only=False)
        count = 0
        for prod in sci:
            if count >= max_files:
                break
            obs_id = prod["obs_id"]
            if obs_id in processed_ids:
                continue
            dest_dir = FITS_DIR / prod["obs_collection"] / obs_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            filename = prod["productFilename"]
            url = prod["dataURI"]
            # MAST uses URIs like mast:HST/product/...; use the download API
            download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={url}"
            download_file(download_url, dest_dir / filename)
            processed_ids.append(obs_id)
            count += 1
        if target_name not in processed:
            processed[target_name] = []
        processed[target_name].extend(obs_id for obs_id in new["obs_id"][:count])
        # Stop after first level with data
        break


def fetch_data() -> None:
    """Entry point to fetch new data for configured targets."""
    ensure_dirs()
    processed = load_processed()
    targets = [
        {"label": "Pillars of Creation", "target": "M16", "collection": "HST", "levels": [2, 1, 0]},
        {"label": "Stephan’s Quintet", "target": "Stephan's Quintet", "collection": "HST", "levels": [2, 1, 0]},
        {"label": "SMACS 0723 Deep Field", "target": "SMACS 0723", "collection": "JWST", "levels": [2, 1, 0]},
    ]
    for t in targets:
        target = t["target"]
        mission = t["collection"]
        levels = t["levels"]
        logger.info("Processing %s", t["label"])
        download_science_fits(target, mission, levels, processed, max_files=2)
        # Download one preview image (calib level 3) per target
        previews = fetch_preview_urls(target, mission, calib_level=3)
        if previews:
            # Save using a simple target‑based filename
            name = target.replace(" ", "_") + "_preview.jpg"
            dest = RENDERED_DIR / name
            if not dest.exists():
                download_file(previews[0], dest)
        else:
            logger.info("No level‑3 preview for %s", t["label"])
    save_processed(processed)
    logger.info("Data fetch completed.")


if __name__ == "__main__":
    fetch_data()