"""
get_mast_science_and_product.py
================================

This script demonstrates how to download both calibrated instrument data
(*science level*) and fully processed preview images (*product level*)
from the Mikulski Archive for Space Telescopes (MAST).  It is designed
for projects that need meaningful raw input (level 2) and a final
science image (level 3) for visualisation and analysis.  In contrast
to earlier attempts that focused on level‑0 or level‑1 data, this
approach targets calibration level 2 exposures, which contain
astronomical structure and variation, and level 3 previews, which
represent the end product.

The script uses `astroquery.mast` to discover observations and
download FITS files, and the public CAOM API to fetch JPEG
previews.  It attempts to select the lowest available calibration
level from a supplied list (by default `[2, 1, 0]`), so it can
gracefully handle missions that do not publish level‑0 or level‑1
products.  If no data exist at any requested level, the script will
report this and skip to the next target.

Usage::

    python get_mast_science_and_product.py

The script will create ``data/fits`` and ``data/rendered`` directories
relative to the current working directory (if they do not exist)
and populate them with downloaded FITS files and preview images.

Dependencies
------------
* Python 3.8+
* requests
* numpy
* astropy
* astroquery

You can install missing dependencies via ``pip install requests numpy
astropy astroquery``.

"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests

try:
    from astroquery.mast import Observations
except ImportError as exc:
    raise SystemExit(
        "astroquery is required for this script. Install it via `pip install astroquery`."
    ) from exc


def fetch_preview_urls(
    target_name: str,
    obs_collection: str,
    calib_level: int,
    pagesize: int = 100,
    max_pages: int = 2,
) -> List[str]:
    """Return a list of JPEG preview URLs for a given target and calibration level.

    This helper function uses the public ``Mast.Caom.Filtered`` service to fetch
    observation metadata and returns any available preview URLs.  Using a GET
    request with a JSON payload encoded in the ``request`` query parameter
    circumvents the 403 errors seen when using POST.

    Parameters
    ----------
    target_name : str
        Name of the object to search for (e.g. 'M16' or 'SMACS 0723').
    obs_collection : str
        Mission collection (e.g. 'HST' or 'JWST').
    calib_level : int
        Calibration level to filter on (0–3).
    pagesize : int, optional
        Maximum number of rows to request per page (default 100).
    max_pages : int, optional
        Maximum number of pages to retrieve (default 2).

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
        The destination path where the image will be saved. Parent
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


def download_science_fits(
    target_name: str,
    obs_collection: str,
    calib_levels: List[int],
    max_files: int = 1,
) -> List[Path]:
    """Download the lowest available calibration FITS file(s) for a target.

    This function queries observations around ``target_name`` in the given
    collection and attempts to find data at the lowest calibration level
    listed in ``calib_levels``.  It uses a generous 0.2° search radius to
    capture off‑axis observations.  Once a level with at least one
    observation is found, it downloads up to ``max_files`` science products.

    Parameters
    ----------
    target_name : str
        Name of the object to search for (e.g. 'M16' or 'SMACS 0723').
    obs_collection : str
        Mission collection (e.g. 'HST' or 'JWST').
    calib_levels : list of int
        Calibration levels to try in order for science data.  For JWST you might
        supply ``[2, 1, 0]`` if level 2 products are available, otherwise
        falling back.
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
    for level in calib_levels:
        sub = obs_table[obs_table["calib_level"] == level]
        if len(sub) > 0:
            selected_level = level
            obs_table = sub
            break
    if selected_level is None:
        levels_str = ", ".join(str(l) for l in calib_levels)
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
    download_dir.mkdir(parents=True, exist_ok=True)
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
    # Configure targets with mission and candidate calibration levels.
    targets: List[Dict[str, object]] = [
        {
            "label": "Pillars of Creation (Eagle Nebula)",
            "target": "M16",
            "collection": "HST",
            "calib_levels": [2, 1, 0],
        },
        {
            "label": "Cosmic Cliffs (Carina Nebula)",
            "target": "NGC 3324",
            "collection": "JWST",
            "calib_levels": [2, 1, 0],
        },
        {
            "label": "Southern Ring Nebula",
            "target": "NGC 3132",
            "collection": "JWST",
            "calib_levels": [2, 1, 0],
        },
        {
            "label": "Stephan’s Quintet",
            "target": "Stephan's Quintet",
            "collection": "HST",
            "calib_levels": [2, 1, 0],
        },
        {
            "label": "Webb’s First Deep Field (SMACS 0723)",
            "target": "SMACS 0723",
            "collection": "JWST",
            "calib_levels": [2, 1, 0],
        },
    ]
    for obj in targets:
        label: str = obj["label"]  # type: ignore[assignment]
        tname: str = obj["target"]  # type: ignore[assignment]
        collection: str = obj["collection"]  # type: ignore[assignment]
        levels: List[int] = obj["calib_levels"]  # type: ignore[assignment]
        print(f"\n=== {label} ===")
        # Download science-level FITS (level 2 preferred)
        science_files = download_science_fits(tname, collection, levels, max_files=1)
        if science_files:
            print(f"Downloaded science-level FITS for {label}: {science_files[0].name}")
        # Fetch level 3 previews (final product)
        previews = fetch_preview_urls(tname, collection, calib_level=3, pagesize=20, max_pages=2)
        if previews:
            preview_url = previews[0]
            # Use target name to build a filename; replace spaces with underscores
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
