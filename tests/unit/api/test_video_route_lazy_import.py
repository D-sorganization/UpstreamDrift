"""Regression tests for issue #2809: video route lazy import.

The slim runtime image intentionally omits ``cv2`` and ``mediapipe``. Before
this change the route module imported ``VideoPosePipeline`` at top level,
which caused the whole route module to fail to import and be skipped by the
route registry, yielding 404s on ``/api/v1/video/*``.

The contract these tests lock in:

1. ``src.api.routes.video`` imports successfully even when ``cv2`` is absent
   at the point of route discovery (no top-level dependency on the pipeline).
2. A router is registered on the module so route discovery picks it up.
3. ``dependencies._load_video_pipeline_classes`` converts ImportError into a 503
   ``HTTPException`` with a clear, actionable message.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest
from fastapi import APIRouter, HTTPException


def test_video_route_module_imports_without_cv2() -> None:
    """Route module must not require cv2 at import time (issue #2809)."""
    # Force reimport under a sys.modules shim that breaks cv2 import.
    module_name = "src.api.routes.video"
    previously_loaded = sys.modules.pop(module_name, None)
    # Also pop the transitive pipeline module so the shim is actually exercised
    # if any code path tries to import it at module load.
    pipeline_name = "src.shared.python.gui_pkg.video_pose_pipeline"
    previously_loaded_pipeline = sys.modules.pop(pipeline_name, None)

    try:
        with mock.patch.dict(sys.modules, {"cv2": None}):
            module = importlib.import_module(module_name)
            assert isinstance(module.router, APIRouter)
    finally:
        # Restore to avoid polluting other tests.
        if previously_loaded is not None:
            sys.modules[module_name] = previously_loaded
        else:
            sys.modules.pop(module_name, None)
            importlib.import_module(module_name)
        if previously_loaded_pipeline is not None:
            sys.modules[pipeline_name] = previously_loaded_pipeline


def test_load_video_pipeline_classes_raises_503_on_missing_dep() -> None:
    """Missing cv2/mediapipe must surface as 503, not a bare ImportError."""
    from src.api import dependencies

    # Drop the cached pipeline module so the guarded import re-runs with the
    # shimmed ``cv2`` and raises ImportError inside the helper.
    sys.modules.pop("src.shared.python.gui_pkg.video_pose_pipeline", None)
    with (
        mock.patch.dict(sys.modules, {"cv2": None}),
        pytest.raises(HTTPException) as excinfo,
    ):
        dependencies._load_video_pipeline_classes()

    assert excinfo.value.status_code == 503
    assert "Video analysis is unavailable" in excinfo.value.detail


def test_video_route_has_expected_endpoints() -> None:
    """Guard against accidental route removal during refactors."""
    from src.api.routes import video as video_module

    paths = {route.path for route in video_module.router.routes}
    assert "/analyze/video" in paths
    assert "/analyze/video/async" in paths
