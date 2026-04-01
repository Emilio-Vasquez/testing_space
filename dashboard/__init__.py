"""Initialize the dashboard package.

This file makes the ``dashboard`` directory a proper Python package so
that relative imports within ``dashboard`` work correctly.  Without
``__init__.py`` Python will treat the directory as a namespace
package, which breaks the relative imports used in ``app.py`` when
registering API blueprints.  See the log messages about routes failing
to register for details.
"""

__all__ = []