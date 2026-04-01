"""API blueprint for live data.

This module defines a Flask blueprint that exposes an endpoint to serve
"live" data for the dashboard.  In the initial demonstration pipeline
the live endpoint merely returned the most recently processed dataset.
However, for the streaming version of the system the pipeline writes
out a file called ``live_raw.json`` containing the latest raw measurements
(e.g. solar wind speed, signal strength, proton density).  The endpoint
first looks for this streaming file and falls back to ``processed.json``
for backward compatibility if the streaming file does not exist.  When
integrating with a true streaming backend, replace the file read logic
with a call to whatever service is providing real‑time data.
"""

from flask import Blueprint, jsonify
from pathlib import Path
import json

# Create the blueprint.  It will be registered on the Flask app in
# ``dashboard/app.py`` if desired.
bp_live = Blueprint('live_api', __name__)

# Determine where processed data resides.  We locate the ``dashboard/data``
# directory relative to this file.  ``__file__`` resolves to
# ``ADPs/dashboard/api/routes_live.py``.  One parent up is
# ``ADPs/dashboard/api`` and its parent is ``ADPs/dashboard``.  The
# ``data`` directory lives under ``dashboard``, so we use parents[1]
# instead of parents[2] (which would incorrectly point to ``ADPs/data``).
data_dir = Path(__file__).resolve().parents[1] / 'data'


@bp_live.route('/api/live')
def live_data() -> "flask.Response":
    """Return the latest live (raw) data as a JSON array.

    The streaming pipeline writes raw telemetry and communication samples
    to ``live_raw.json`` in the ``dashboard/data`` directory.  Clients
    fetch this endpoint periodically to update charts.  If the streaming
    file does not exist, the endpoint falls back to ``processed.json``
    for backward compatibility with the batch pipeline.  When neither
    file exists, an empty list is returned.  The timestamp values are
    returned as strings (ISO 8601 format).
    """
    try:
        # Prefer the live raw file when available
        file_path = data_dir / 'live_raw.json'
        if not file_path.exists():
            file_path = data_dir / 'processed.json'
        if file_path.exists():
            data = json.loads(file_path.read_text())
        else:
            data = []
    except Exception:
        data = []
    return jsonify(data)