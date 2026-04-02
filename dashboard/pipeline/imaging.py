from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from .catalog import get_entry

ImageArray = np.ndarray


def find_best_image_hdu(hdul: fits.HDUList) -> Optional[Tuple[int, str, ImageArray]]:
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
def load_science_image(slug: str) -> Optional[Tuple[ImageArray, Dict[str, object]]]:
    entry = get_entry(slug)
    if entry is None:
        return None
    _title, path = entry
    try:
        with fits.open(path, memmap=False) as hdul:
            result = find_best_image_hdu(hdul)
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


def compute_valid_crop_bounds(arr: ImageArray, eps: float = 1e-8) -> Optional[Tuple[int, int, int, int]]:
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


def crop_with_bounds(arr: ImageArray, bounds: Optional[Tuple[int, int, int, int]]) -> ImageArray:
    if bounds is None:
        return arr
    r0, r1, c0, c1 = bounds
    return arr[r0:r1, c0:c1]


def normalize_to_uint8(arr: ImageArray) -> ImageArray:
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
    clipped = np.clip(np.nan_to_num(arr, nan=vmin, posinf=vmax, neginf=vmin), vmin, vmax)
    scaled = (clipped - vmin) / (vmax - vmin)
    stretched = np.arcsinh(12.0 * scaled) / np.arcsinh(12.0)
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


def compute_histogram(values: ImageArray, bins: int = 20, value_range: Tuple[float, float] = (0, 255)) -> List[int]:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    return hist.astype(int).tolist()
