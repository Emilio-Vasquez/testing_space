"""API blueprint for processed data.

This blueprint exposes an endpoint that returns the full set of processed
feature data.  The processed dataset may be large, so applications may
choose to paginate or summarise it.  In this simple demonstration the
entire JSON file is returned to the client.
"""

from flask import Blueprint, jsonify
from pathlib import Path
import json

bp_processed = Blueprint('processed_api', __name__)

# The ``data`` directory resides under the ``dashboard`` package.  Use
# parents[1] to navigate to ``ADPs/dashboard`` and append ``data``.
data_dir = Path(__file__).resolve().parents[1] / 'data'

@bp_processed.route('/api/processed_full')
def processed_full():
    """Return the entire processed dataset.

    The pipeline writes its processed output to ``processed.json``.
    This endpoint returns all records.  For performance reasons, the
    dashboard uses ``/api/processed`` to request only the latest 500
    records; this endpoint is provided for completeness.
    """
    try:
        file_path = data_dir / 'processed.json'
        if file_path.exists():
            data = json.loads(file_path.read_text())
        else:
            data = []
    except Exception:
        data = []
    return jsonify(data)