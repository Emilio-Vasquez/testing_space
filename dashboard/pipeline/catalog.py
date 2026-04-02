from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import current_app, url_for


def resolve_data_base(current_dir: Path) -> Path:
    base_candidates = [
        current_dir / "data",
        current_dir.parents[1] / "data" if len(current_dir.parents) > 1 else None,
        current_dir.parents[2] / "data" if len(current_dir.parents) > 2 else None,
    ]
    base_candidates = [candidate for candidate in base_candidates if candidate is not None]
    for candidate in base_candidates:
        if (candidate / "fits" / "mastDownload").exists():
            return candidate
    raise RuntimeError("Cannot locate data directory. Ensure data/fits/mastDownload exists.")


def discover_raw_files(fits_base: Path, slug_title_overrides: Dict[str, str]) -> Dict[str, Tuple[str, Path]]:
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
        title = slug_title_overrides.get(slug, slug.replace("_", " ").title())
        mapping[slug] = (title, file)
    return mapping


def get_raw_objects() -> Dict[str, Tuple[str, Path]]:
    return current_app.config["RAW_OBJECTS"]


def get_rendered_base() -> Path:
    return current_app.config["RENDERED_BASE"]


def build_nav() -> List[Dict[str, str]]:
    raw_objects = get_raw_objects()
    preferred_order = current_app.config["PREFERRED_NAV_ORDER"]
    nav_items: List[Dict[str, str]] = []
    for slug in preferred_order:
        if slug in raw_objects:
            title, _path = raw_objects[slug]
            nav_items.append({"slug": slug, "title": title})
    for slug, (title, _path) in raw_objects.items():
        if slug not in preferred_order:
            nav_items.append({"slug": slug, "title": title})
    return nav_items


def load_rendered_images() -> List[Dict[str, str]]:
    rendered_base = get_rendered_base()
    images: List[Dict[str, str]] = []
    if rendered_base.exists():
        for img in sorted(rendered_base.iterdir()):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                images.append({
                    "filename": img.name,
                    "label": img.stem.replace("_", " ").title(),
                    "url": url_for("pages_bp.serve_rendered", filename=img.name),
                })
    return images


def get_entry(slug: str) -> Optional[Tuple[str, Path]]:
    return get_raw_objects().get(slug)
