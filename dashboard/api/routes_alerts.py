"""API blueprint for anomaly alerts.

This module provides an endpoint that returns the list of generated
alerts.  Alerts are written by the pipeline to ``alerts.json`` (for
normal runs) and ``alerts_attack.json`` (for attack runs).  The API
returns both sets combined, tagging attack alerts with a flag for the
frontend to distinguish them.
"""

from flask import Blueprint, jsonify
from pathlib import Path
import json

bp_alerts = Blueprint('alerts_api', __name__)

# The ``data`` directory is located under ``dashboard``.  Use
# parents[1] to reach ``ADPs/dashboard`` instead of parents[2] (which
# points to the project root) and append ``data``.
data_dir = Path(__file__).resolve().parents[1] / 'data'


def _read_alert_file(path: Path, attack: bool = False) -> list:
    """Read a YAML or JSON file of alerts and mark attack status.

    Parameters
    ----------
    path : Path
        Path to the alerts file.  The file may be JSON or YAML.
    attack : bool
        Whether the alerts originate from an attack scenario.

    Returns
    -------
    list
        List of alert dictionaries with an additional ``attack`` key.
    """
    if not path.exists():
        return []
    try:
        # Files are written as YAML using ``yaml.dump`` but JSON is
        # backwards compatible; attempt JSON load first
        text = path.read_text()
        try:
            alerts = json.loads(text)
        except json.JSONDecodeError:
            import yaml  # defer import to avoid heavy dependency if unused
            alerts = yaml.safe_load(text) or []
    except Exception:
        alerts = []
    # Tag each alert with attack flag
    for a in alerts:
        a['attack'] = attack
    return alerts


@bp_alerts.route('/api/alerts_all')
def alerts():
    """Return all alerts from normal and attack runs.

    This endpoint reads both ``alerts.json`` and ``alerts_attack.json`` if
    present and concatenates them.  The ``attack`` key in the resulting
    dictionaries indicates whether the alert originated from an attack
    scenario.  Clients should filter or style alerts accordingly.
    """
    normal_alerts = _read_alert_file(data_dir / 'alerts.json', attack=False)
    attack_alerts = _read_alert_file(data_dir / 'alerts_attack.json', attack=True)
    return jsonify(normal_alerts + attack_alerts)