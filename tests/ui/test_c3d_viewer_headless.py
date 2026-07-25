"""Headless smoke test for C3D Viewer."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


_VIEWER_SUFFIX = (
    "engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps.c3d_viewer"
)
_CACHED_VIEWER: list[types.ModuleType | None] = []


# Handle import of module with invalid identifier (3D_Golf_Model)
def import_c3d_viewer() -> types.ModuleType | None:
    """Import the C3D viewer module dynamically.

    Memoized: the module defines ``QWidget`` subclasses, and ``patch.dict`` on
    ``sys.modules`` evicts it on teardown. Re-importing then fails with sip's
    "cannot load module more than once per process", so hold the first
    successful import for the lifetime of the test session.
    """
    if _CACHED_VIEWER:
        return _CACHED_VIEWER[0]
    module: types.ModuleType | None = None
    for module_name in (f"src.{_VIEWER_SUFFIX}", _VIEWER_SUFFIX):
        try:
            module = importlib.import_module(module_name)
            break
        except ImportError:
            continue
    _CACHED_VIEWER.append(module)
    return module


@pytest.mark.skipif(sys.platform == "linux", reason="Requires X11 or Xvfb on Linux")
def test_c3d_viewer_instantiation(qtbot) -> None:
    """Test that the main window can be instantiated without crashing."""
    with patch.dict(sys.modules, {"c3d_reader": MagicMock()}):
        c3d_viewer = import_c3d_viewer()

        if c3d_viewer is None:
            pytest.skip("Could not import c3d_viewer due to path issues")

        # Now instantiate
        window = c3d_viewer.C3DViewerMainWindow()

        # qtbot.addWidget may fail with dynamically imported QWidgets
        # from non-standard module paths (pytest-qt type check limitation)
        try:
            qtbot.addWidget(window)
        except TypeError:
            # Manually ensure cleanup
            window.close()
            window.deleteLater()

        assert window.windowTitle() == "C3D Motion Analysis Viewer"
        assert window.model is None

        # Verify tabs exist
        central_widget = window.centralWidget()
        assert central_widget is not None
        if hasattr(central_widget, "count"):
            assert central_widget.count() >= 1


# ----- shared security import resilience (issue #8073) -----------------------


@pytest.mark.unit
def test_resolve_validate_path_finds_shared_helper() -> None:
    """The viewer resolves ``validate_path`` under at least one import root.

    The file-open handler previously imported
    ``shared.python.security.security_utils`` unconditionally. Under the
    embedded launcher tab that name resolves against the sibling Tools
    checkout, so the import raised ``ModuleNotFoundError`` from inside the
    handler and Qt's crash dialog took the whole launcher down with it.
    """
    with patch.dict(sys.modules, {"c3d_reader": MagicMock()}):
        c3d_viewer = import_c3d_viewer()
    if c3d_viewer is None:
        pytest.skip("Could not import c3d_viewer due to path issues")

    resolved = c3d_viewer._resolve_validate_path()

    assert resolved is not None
    assert callable(resolved)


@pytest.mark.unit
def test_resolve_validate_path_returns_none_instead_of_raising() -> None:
    """When no candidate imports, the resolver degrades instead of crashing."""
    with patch.dict(sys.modules, {"c3d_reader": MagicMock()}):
        c3d_viewer = import_c3d_viewer()
    if c3d_viewer is None:
        pytest.skip("Could not import c3d_viewer due to path issues")

    def _always_missing(name: str) -> types.ModuleType:
        raise ImportError(f"No module named {name!r}")

    with patch.object(c3d_viewer.importlib, "import_module", _always_missing):
        assert c3d_viewer._resolve_validate_path() is None


@pytest.mark.unit
def test_load_c3d_file_from_path_reports_missing_security_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing security helper yields a modal, not an unhandled ImportError."""
    with patch.dict(sys.modules, {"c3d_reader": MagicMock()}):
        c3d_viewer = import_c3d_viewer()
    if c3d_viewer is None:
        pytest.skip("Could not import c3d_viewer due to path issues")

    from PyQt6 import QtWidgets

    owner = next(
        cls
        for cls in vars(c3d_viewer).values()
        if isinstance(cls, type) and "load_c3d_file_from_path" in vars(cls)
    )

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        staticmethod(lambda *args, **kwargs: shown.append((args[1], args[2]))),
    )
    monkeypatch.setattr(c3d_viewer, "_resolve_validate_path", lambda: None)

    # Call the handler against a stand-in ``self``: the guarded branch only
    # needs a message-box parent, so constructing the full GUI (matplotlib
    # canvases, 3D viewport) would add nothing but flakiness.
    owner.load_c3d_file_from_path(object(), "whatever.c3d")

    assert shown, "expected a user-facing dialog"
    title, body = shown[0]
    assert title == "Cannot open C3D file"
    assert "PYTHONPATH" in body
