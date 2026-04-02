from __future__ import annotations

"""
Pipeline-centered Flask application for the Astrophysics Cybersecurity dashboard.

This refactor preserves the current UI behavior while reorganizing the codebase
around pipeline stages:
- acquisition/catalog discovery
- raw ingestion and monitoring
- attack application
- anomaly detection
- containment and restoration
- Raspberry Pi attack session management
"""

from pathlib import Path
from typing import Dict

from flask import Flask

from dashboard.pipeline.catalog import discover_raw_files, resolve_data_base
from dashboard.routes import attack_api_bp, monitoring_api_bp, pages_bp


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    current_dir = Path(__file__).resolve().parent
    data_base = resolve_data_base(current_dir)
    fits_base = data_base / "fits" / "mastDownload"
    rendered_base = data_base / "rendered"
    slug_title_overrides: Dict[str, str] = {
        "ice00ap1q_raw": "Pillars of Creation (M16)",
        "hst_8063_p7_nic_nic3_f110w_01_drz": "Stephan’s Quintet",
        "jw02736006001_02101_00001_nrs1_cal": "SMACS 0723 Deep Field (JWST)",
    }
    preferred_nav_order = [
        "hst_8063_p7_nic_nic3_f110w_01_drz",
        "jw02736006001_02101_00001_nrs1_cal",
        "ice00ap1q_raw",
    ]
    app.config.update(
        DATA_BASE=data_base,
        FITS_BASE=fits_base,
        RENDERED_BASE=rendered_base,
        SLUG_TITLE_OVERRIDES=slug_title_overrides,
        PREFERRED_NAV_ORDER=preferred_nav_order,
        RAW_OBJECTS=discover_raw_files(fits_base, slug_title_overrides),
    )
    app.register_blueprint(pages_bp)
    app.register_blueprint(monitoring_api_bp)
    app.register_blueprint(attack_api_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
