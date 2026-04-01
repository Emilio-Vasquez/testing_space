import json
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import unquote
import re

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


API_URL = "https://mast.stsci.edu/api/v0/invoke"
DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"


def mast_query(request_obj):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    resp = requests.post(
        API_URL,
        data="request=" + json.dumps(request_obj),
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def safe_name_from_product_row(row, fallback="product.fits"):
    # Prefer productFilename if present
    filename = str(row.get("productFilename", "")).strip()
    if filename:
        return re.sub(r'[<>:"/\\\\|?*]', "_", filename)

    uri = unquote(str(row.get("dataURI", "")).strip())

    # HLA-style URI with ?dataset=...
    m = re.search(r"dataset=([^&]+)", uri)
    if m:
        return re.sub(r'[<>:"/\\\\|?*]', "_", m.group(1))

    tail = Path(uri).name
    tail = re.sub(r'[<>:"/\\\\|?*]', "_", tail)
    return tail or fallback


def first_fits_product_for_obsid(obsid):
    req = {
        "service": "Mast.Caom.Products",
        "params": {"obsid": int(obsid)},
        "format": "json",
    }
    result = mast_query(req)
    rows = result.get("data", [])
    if not rows:
        return None

    pdf = pd.DataFrame(rows)

    # Keep FITS only
    if "productFilename" in pdf.columns:
        mask = pdf["productFilename"].astype(str).str.lower().str.endswith(".fits")
        pdf = pdf[mask].copy()
    else:
        mask = pdf["dataURI"].astype(str).str.lower().str.contains(r"\.fits($|[?&])", regex=True)
        pdf = pdf[mask].copy()

    if pdf.empty:
        return None

    # Prefer SCIENCE products when available
    if "productType" in pdf.columns:
        sci = pdf[pdf["productType"].astype(str).str.upper() == "SCIENCE"]
        if not sci.empty:
            pdf = sci

    return pdf.iloc[0].to_dict()


def download_product(product_row, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    uri = product_row["dataURI"]
    filename = safe_name_from_product_row(product_row)
    out_path = out_dir / filename

    resp = requests.get(DOWNLOAD_URL, params={"uri": uri}, timeout=300)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(resp.content)

    return out_path


def extract_first_image_array(fits_path):
    with fits.open(fits_path) as hdul:
        for i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is None:
                continue

            arr = np.array(data)

            # direct 2D image
            if arr.ndim == 2:
                return i, arr

            # some products have extra dims; grab first plane if sensible
            if arr.ndim >= 3:
                squeezed = np.squeeze(arr)
                if squeezed.ndim == 2:
                    return i, squeezed

    return None, None


def render_fits_image(fits_path, output_png):
    hdu_index, arr = extract_first_image_array(fits_path)
    if arr is None:
        return False, None

    arr = np.array(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False, None

    vmin = np.percentile(finite, 5)
    vmax = np.percentile(finite, 99)

    plt.figure(figsize=(7, 7))
    plt.imshow(arr, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(f"{fits_path.name} (HDU {hdu_index})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    return True, hdu_index


# -----------------------------
# 1) Query observations
# -----------------------------
# Keep this constrained so it doesn't explode in size.
# You can change proposal_id / target_name / instrument_name as needed.
obs_request = {
    "service": "Mast.Caom.Filtered",
    "format": "json",
    "pagesize": 200,
    "page": 1,
    "params": {
        "columns": ",".join([
            "obsid",
            "obs_id",
            "target_name",
            "obs_collection",
            "instrument_name",
            "proposal_id",
            "calib_level",
            "dataproduct_type",
            "dataRights",
            "t_min"
        ]),
        "filters": [
            {"paramName": "obs_collection", "values": ["HST"]},
            {"paramName": "dataRights", "values": ["PUBLIC"]},
            {"paramName": "proposal_id", "values": ["12062"]},
            {"paramName": "dataproduct_type", "values": ["image"]},
        ],
    },
}

obs_result = mast_query(obs_request)
obs_rows = obs_result.get("data", [])
obs_df = pd.DataFrame(obs_rows)

if obs_df.empty:
    raise RuntimeError("No observations found.")

print("Available calibration levels:", sorted(obs_df["calib_level"].dropna().unique().tolist()))
print(obs_df[["obsid", "obs_id", "target_name", "instrument_name", "calib_level"]].head(10))


# -----------------------------
# 2) For each calibration level, get one FITS product
# -----------------------------
base_dir = Path("data/stage_demo")
fits_dir = base_dir / "fits"
png_dir = base_dir / "rendered"
fits_dir.mkdir(parents=True, exist_ok=True)
png_dir.mkdir(parents=True, exist_ok=True)

summary = []

for level in sorted(obs_df["calib_level"].dropna().unique().tolist()):
    level_rows = obs_df[obs_df["calib_level"] == level].copy()

    if level_rows.empty:
        continue

    chosen = level_rows.iloc[0].to_dict()
    obsid = chosen["obsid"]

    print(f"\nTrying calib_level={level} using obsid={obsid} obs_id={chosen['obs_id']}")

    product = first_fits_product_for_obsid(obsid)
    if product is None:
        print(f"No FITS product found for level {level}")
        summary.append({
            "calib_level": level,
            "obsid": obsid,
            "obs_id": chosen["obs_id"],
            "fits_downloaded": False,
            "rendered_image": False,
            "note": "No FITS product found"
        })
        continue

    try:
        level_fits_dir = fits_dir / f"level_{level}"
        level_png_dir = png_dir / f"level_{level}"
        level_fits_dir.mkdir(parents=True, exist_ok=True)
        level_png_dir.mkdir(parents=True, exist_ok=True)

        fits_path = download_product(product, level_fits_dir)
        png_path = level_png_dir / (fits_path.stem + ".png")

        rendered, hdu_index = render_fits_image(fits_path, png_path)

        print("Downloaded:", fits_path)
        print("Rendered:", rendered, "HDU:", hdu_index)

        summary.append({
            "calib_level": level,
            "obsid": obsid,
            "obs_id": chosen["obs_id"],
            "fits_downloaded": True,
            "rendered_image": rendered,
            "fits_path": str(fits_path),
            "png_path": str(png_path) if rendered else "",
            "note": f"Rendered from HDU {hdu_index}" if rendered else "FITS downloaded but not image-like"
        })

    except Exception as e:
        print(f"Failed at level {level}: {e}")
        summary.append({
            "calib_level": level,
            "obsid": obsid,
            "obs_id": chosen["obs_id"],
            "fits_downloaded": False,
            "rendered_image": False,
            "note": str(e)
        })

summary_df = pd.DataFrame(summary)
print("\nSummary:")
print(summary_df)