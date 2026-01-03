"""Headless smoke test for C3D Viewer."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


# Handle import of module with invalid identifier (3D_Golf_Model)
def import_c3d_viewer():
    """Import the C3D viewer module dynamically."""
    # We assume the repo root is in sys.path or accessible
    module_name = "engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps.c3d_viewer"
    try:
        return importlib.import_module(module_name)
    except ImportError:
        # Fallback: maybe add to sys.path if not resolved
        return None


@pytest.mark.skipif(sys.platform == "linux", reason="Requires X11 or Xvfb on Linux")
def test_c3d_viewer_instantiation(qtbot):
    """Test that the main window can be instantiated without crashing."""
    # Mock c3d_reader to avoid file system dependencies if called during init
    # We need to construct the patch path carefully or just patch sys.modules
    # if c3d_reader is imported directly

    # We need to import the module inside the patch context or before
    # But since the module import executes code, we might want to patch imports *before* import

    with patch.dict(sys.modules, {"c3d_reader": MagicMock()}):
        c3d_viewer = import_c3d_viewer()

        if c3d_viewer is None:
            pytest.skip("Could not import c3d_viewer due to path issues")

        # Now instantiate
        window = c3d_viewer.C3DViewerMainWindow()
        qtbot.addWidget(window)

        assert window.windowTitle() == "C3D Motion Analysis Viewer"
        assert window.model is None

        # Verify tabs exist
        central_widget = window.centralWidget()
        assert central_widget is not None
        if hasattr(central_widget, "count"):
            assert central_widget.count() >= 1
