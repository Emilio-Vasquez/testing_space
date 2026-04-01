"""Initialize the API subpackage for the dashboard.

This file marks ``api`` as a Python package so that the relative
imports in ``dashboard/app.py`` (e.g. ``from .api.routes_live
import bp_live``) resolve correctly.  Without this file, the Flask app
would be unable to import and register the API blueprints, leading to
404 errors on endpoints such as ``/api/live``.
"""

__all__ = ['routes_live', 'routes_processed', 'routes_alerts']