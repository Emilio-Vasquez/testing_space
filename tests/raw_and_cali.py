"""
download_raw_and_calibrated_images_fixed.py
==========================================

This script is a robust replacement for the earlier ``download_raw_and_calibrated_images.py``
that failed to find data for several targets.  It demonstrates how to download
both *raw* instrument data (or the nearest available calibration level) and
a *calibrated* preview image from the Mikulski Archive for Space Telescopes
(MAST) for a handful of iconic objects.  It addresses a number of issues
encountered by the original script:

* Some objects have **no level‑0 data** published.  JWST Early Release
  Observations (Carina Nebula, Southern Ring Nebula, SMACS 0723) typically
  start at level 2; likewise many HST datasets are archived at level 1.  This
  script takes a list of candidate calibration levels (e.g. ``[0, 1, 2]``)
  per target and downloads the lowest available level.  If no data exists
  at any of the requested levels, it reports that fact and moves on.

* The original script sent a POST request to the MAST API, which is blocked
  for unauthenticated users.  The ``fetch_preview_urls`` function now uses
  a GET request with a ``request`` query parameter; this matches the
  recommended usage for the ``Mast.Caom.Filtered`` service.

* Search radius and calibration level filtering are relaxed.  A 0.2° radius
  is used instead of 0.05°, increasing the likelihood of capturing off‑axis
  observations.  When scanning for raw data, the script stops at the
  first calibration level that produces any observations.

The script saves downloaded FITS files into ``data/fits`` and preview
JPEGs into ``data/rendered``.  At the end it prints a summary of files.

Requirements
------------
This script requires Python 3.8+, ``requests``, ``numpy``, ``astropy``, and
``astroquery``.  Internet access is required to download data from MAST.

Usage
-----
Run the script from the project root:

.. code-block:: bash

    python download_raw_and_calibrated_images_fixed.py

It will create the necessary directories and download files for each
configured target.

"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests

try:
    from astroquery.mast import Observations
except ImportError as exc:
    raise SystemExit(
        "astroquery is required for this script.  Install it via `pip install astroquery`."
    ) from exc

from astropy.io import fits  # noqa: F401  # imported for side effect (fits utils may be used by callers)


def fetch_preview_urls(
    target_name: str,
    obs_collection: str,
    calib_level: int,
    pagesize: int = 100,
    max_pages: int = 1,
) -> List[str]:
    """Return a list of JPEG preview URLs for a given target and calibration level.

    This helper function uses the public ``Mast.Caom.Filtered`` service to fetch
    observation metadata and returns any available preview URLs.  Using a GET
    request with a JSON payload encoded in the ``request`` query parameter
    circumvents the 403 errors seen when using POST.

    Parameters
    ----------
    target_name : str
        Name of the object to search for (e.g. ``'NGC 3324'`` or ``'SMACS 0723'``).
    obs_collection : str
        Mission collection (e.g. ``'HST'`` or ``'JWST'``).
    calib_level : int
        Calibration level to filter on (0–3).
    pagesize : int, optional
        Maximum number of rows to request per page (default 100).
    max_pages : int, optional
        Maximum number of pages to retrieve (default 1).

    Returns
    -------
    List[str]
        A list of JPEG preview URLs.  If no previews are available, the list
        will be empty.
    """
    service_url = "https://mast.stsci.edu/api/v0/invoke"
    preview_urls: List[str] = []
    for page in range(1, max_pages + 1):
        payload = {
            "service": "Mast.Caom.Filtered",
            "params": {
                "columns": "jpegURL",
                "filters": [
                    {"paramName": "obs_collection", "values": [obs_collection]},
                    {"paramName": "target_name", "values": [target_name]},
                    {"paramName": "calib_level", "values": [calib_level]},
                ],
            },
            "format": "json",
            "pagesize": pagesize,
            "page": page,
        }
        try:
            resp = requests.get(service_url, params={"request": json.dumps(payload)}, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            print(f"Failed to fetch previews for {target_name}: {exc}")
            return []
        data = resp.json().get("data", [])
        if not data:
            break
        for row in data:
            url = row.get("jpegURL")
            if url:
                preview_urls.append(url)
        if len(data) < pagesize:
            break
    return preview_urls


def download_preview(preview_url: str, output_path: Path) -> None:
    """Download a JPEG preview image from MAST.

    Parameters
    ----------
    preview_url : str
        The URL to a JPEG preview obtained from MAST metadata.
    output_path : pathlib.Path
        The destination path where the image will be saved.  Parent
        directories must already exist.
    """
    try:
        resp = requests.get(preview_url, stream=True, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Failed to download {preview_url}: {exc}")
        return
    with output_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    print(f"Saved preview to {output_path}")


def download_raw_fits(
    target_name: str,
    obs_collection: str,
    raw_levels: List[int],
    max_files: int = 1,
) -> List[Path]:
    """Download the lowest available calibration FITS file(s) for a target.

    This function queries observations around ``target_name`` in the given
    collection and attempts to find data at the lowest calibration level
    listed in ``raw_levels``.  It uses a generous 0.2° search radius to
    capture off‑axis observations.  Once a level with at least one
    observation is found, it downloads up to ``max_files`` science products.

    Parameters
    ----------
    target_name : str
        Name of the object to search for (e.g. ``'M16'`` or ``'SMACS 0723'``).
    obs_collection : str
        Mission collection (e.g. ``'HST'`` or ``'JWST'``).
    raw_levels : list of int
        Calibration levels to try in order for raw data.  For JWST you might
        supply ``[1, 2]`` if level 0 products are unavailable.
    max_files : int, optional
        Maximum number of FITS files to download (default 1).

    Returns
    -------
    list of pathlib.Path
        A list of paths to the downloaded FITS files.  Empty if none found.
    """
    print(f"Searching for observations of {target_name} in {obs_collection}…")
    try:
        obs_table = Observations.query_object(target_name, radius="0.2 deg")
    except Exception as exc:
        print(f"Query failed for {target_name}: {exc}")
        return []
    obs_table = obs_table[obs_table["obs_collection"] == obs_collection]
    if len(obs_table) == 0:
        print(f"No observations found for {target_name} in {obs_collection}.")
        return []
    selected_level: Optional[int] = None
    for level in raw_levels:
        sub = obs_table[obs_table["calib_level"] == level]
        if len(sub) > 0:
            selected_level = level
            obs_table = sub
            break
    if selected_level is None:
        levels_str = ", ".join(str(l) for l in raw_levels)
        print(f"No observations at calibration levels {levels_str} for {target_name} in {obs_collection}.")
        return []
    print(f"Found {len(obs_table)} observation(s) at calibration level {selected_level}.")
    products = Observations.get_product_list(obs_table)
    science_products = Observations.filter_products(
        products,
        productType="SCIENCE",
        mrp_only=False,
    )
    if len(science_products) == 0:
        print(f"No science products available for {target_name} at level {selected_level}.")
        return []
    to_download = science_products[:max_files]
    download_dir = Path("data/fits")
    download_table = Observations.download_products(
        to_download,
        mrp_only=False,
        download_dir=str(download_dir),
        cache=False,
    )
    downloaded_paths: List[Path] = []
    for row in download_table:
        local = row.get("Local Path")
        if local:
            downloaded_paths.append(Path(local))
    return downloaded_paths


def main() -> None:
    fits_dir = Path("data/fits")
    image_dir = Path("data/rendered")
    fits_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    # Configure targets with mission and candidate raw levels.
    targets: List[Dict[str, object]] = [
        {
            "label": "Pillars of Creation (Eagle Nebula)",
            "target": "M16",
            "collection": "HST",
            "raw_levels": [0, 1],
        },
        {
            "label": "Cosmic Cliffs (Carina Nebula)",
            "target": "NGC 3324",
            "collection": "JWST",
            "raw_levels": [1, 2],
        },
        {
            "label": "Southern Ring Nebula",
            "target": "NGC 3132",
            "collection": "JWST",
            "raw_levels": [1, 2],
        },
        {
            "label": "Stephan’s Quintet",
            "target": "Stephan's Quintet",
            "collection": "HST",
            "raw_levels": [0, 1, 2],
        },
        {
            "label": "Webb’s First Deep Field (SMACS 0723)",
            "target": "SMACS 0723",
            "collection": "JWST",
            "raw_levels": [1, 2],
        },
    ]
    for obj in targets:
        label = obj["label"]
        tname = obj["target"]
        collection = obj["collection"]
        raw_levels = obj["raw_levels"]
        print(f"\n=== {label} ===")
        # Download raw‑level FITS (or nearest) files
        raw_files = download_raw_fits(tname, collection, raw_levels, max_files=1)
        if raw_files:
            print(f"Downloaded raw-level FITS for {label}: {raw_files[0].name}")
        # Fetch level 3 previews
        previews = fetch_preview_urls(tname, collection, calib_level=3, pagesize=20, max_pages=2)
        if previews:
            preview_url = previews[0]
            image_name = f"{tname.replace(' ', '_')}_calib3_preview.jpg"
            output_path = image_dir / image_name
            download_preview(preview_url, output_path)
        else:
            print(f"No level‑3 preview available for {label}.")
    # Print summary
    print("\nSummary of downloaded files:")
    print(f"FITS directory: {fits_dir.resolve()}")
    for f in fits_dir.glob("*.fits"):
        print(f"  {f.name}")
    print(f"Rendered images directory: {image_dir.resolve()}")
    for img in image_dir.glob("*.jpg"):
        print(f"  {img.name}")


if __name__ == "__main__":
    main()