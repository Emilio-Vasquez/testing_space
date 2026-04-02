from __future__ import annotations

from flask import Blueprint, render_template, send_from_directory

from dashboard.pipeline.catalog import build_nav, get_entry, get_rendered_base, load_rendered_images

pages_bp = Blueprint("pages_bp", __name__)


@pages_bp.route("/")
def home():
    return render_template("index.html", images=load_rendered_images(), current_year=2026, nav_items=build_nav())


@pages_bp.route("/rendered/<path:filename>")
def serve_rendered(filename: str):
    return send_from_directory(get_rendered_base(), filename)


@pages_bp.route("/raw/<slug>")
def raw_page(slug: str):
    entry = get_entry(slug)
    if entry is None:
        return render_template("error.html", message="Unknown raw dataset"), 404
    title, _path = entry
    return render_template("raw.html", title=title, slug=slug, nav_items=build_nav(), current_year=2026)
