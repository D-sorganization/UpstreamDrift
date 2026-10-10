"""Regression test for issue #8943's lazy-pandas rule on the Reports routes.

Mirrors ``test_launch_monitor_analytics_lazy_import.py`` for the sibling
module ``src/api/routes/launch_monitor_analytics_reports.py`` (issue #11987
slice 7a): the route registry imports every routes module at API startup, so
a top-level ``import pandas`` (direct, or via the ``reporting`` module it
calls into) would make every API boot pay the pandas import cost. Both
handlers defer ``import pandas`` and the ``reporting`` import into their own
bodies.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest
from fastapi import APIRouter


@pytest.mark.unit
def test_launch_monitor_analytics_reports_imports_without_pandas() -> None:
    """Route module must not require pandas at import time (issue #8943)."""
    module_name = "src.api.routes.launch_monitor_analytics_reports"
    previously_loaded = sys.modules.pop(module_name, None)
    try:
        with mock.patch.dict(sys.modules, {"pandas": None}):
            module = importlib.import_module(module_name)
            assert isinstance(module.router, APIRouter)
    finally:
        if previously_loaded is not None:
            sys.modules[module_name] = previously_loaded
        else:
            sys.modules.pop(module_name, None)
            importlib.import_module(module_name)
