"""Regression test for issue #8943's lazy-pandas rule on the Sessions routes.

Mirrors ``test_launch_monitor_analytics_reports_lazy_import.py`` for the
sibling module ``src/api/routes/launch_monitor_analytics_sessions.py`` (issue
#11987): the route registry imports every routes module at API startup, so a
top-level ``import pandas`` (direct, or via ``import_review``/
``launch_monitor_model``, both of which import pandas) would make every API
boot pay the pandas import cost. Both handlers defer those imports into their
own bodies.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest
from fastapi import APIRouter


@pytest.mark.unit
def test_launch_monitor_analytics_sessions_imports_without_pandas() -> None:
    """Route module must not require pandas at import time (issue #8943)."""
    module_name = "src.api.routes.launch_monitor_analytics_sessions"
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
