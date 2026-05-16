"""Regression tests for issue #5624 — Sidekick must not float.

Defect B: PR #5613 attached the Sidekick via ``addDockWidget`` on a
frameless ``QMainWindow``. The dock's native geometry depends on the
OS window frame that ``FramelessWindowHint`` removes, so the dock
defaults to floating as a sibling top-level window — visible in the
screenshot as a separate "Tools" panel to the right of the main
launcher window.

These tests pin the contract: the Sidekick widget is a third pane of
the main horizontal splitter, never a ``QDockWidget`` and never a
floating top-level window.

Design: TDD/DbC/LOD/DRY.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication, QSplitter


pytestmark = pytest.mark.ui


# The ``ui_setup`` fixture is provided by tests/unit/launcher/conftest.py.
from tests.unit.launcher.test_layout_hierarchy import _outer_vbox  # noqa: E402


@pytest.fixture
def ui_setup_with_sidekick(ui_setup):  # noqa: F811 - intentional fixture chain
    """Mount a fake Sidekick install so ``_install_sidekick_sidebar`` runs.

    The real Sidekick package is heavy and pulls FastAPI + Sidekick state
    storage in.  For these layout tests we only need a bare ``QWidget``
    that survives the install pathway.
    """
    from PyQt6.QtWidgets import QWidget

    # Find the splitter that anchors the main layout (the only direct
    # QSplitter child of outer_vbox).
    layout = _outer_vbox(ui_setup)
    splitter: QSplitter | None = None
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        if isinstance(w, QSplitter):
            splitter = w
            break
    assert splitter is not None, "outer_vbox must contain a QSplitter"

    # Run the production install path.  The launcher's host method
    # ``_install_sidekick_sidebar`` is now expected to embed the widget
    # in the splitter directly, NOT via addDockWidget.  We patch out the
    # real Sidekick factory so we can keep this fast and dependency-free.
    from src.launchers import golf_launcher as _gl

    class _FakeSidebar(QWidget):
        """Stand-in for ``UnifiedToolsSidebar`` — bare QWidget."""

        def __init__(self, *_a, **_k) -> None:
            super().__init__()
            self.setObjectName("fakeSidekickSidebar")

    fake_factory_calls: list[dict] = []

    def _fake_create_tools_sidebar(**kwargs):
        fake_factory_calls.append(kwargs)
        return _FakeSidebar(parent=kwargs.get("parent"))

    # Inject our fake into the integration shim module's namespace so the
    # production code path sees it.
    from src.shared.python.gui_launcher import (
        tools_sidebar_integration as _shim,
    )

    # Build a fake module that exposes create_tools_sidebar.
    import sys
    import types

    fake_mod = types.ModuleType("fake_sidekick_mod")
    fake_mod.create_tools_sidebar = _fake_create_tools_sidebar  # type: ignore[attr-defined]
    sys.modules["fake_sidekick_mod"] = fake_mod

    original_import = _shim._import_sidebar_module
    _shim._import_sidebar_module = lambda: fake_mod  # type: ignore[assignment]
    try:
        # Call the host install method.
        ui_setup._install_sidekick_sidebar()
        yield ui_setup, splitter, fake_factory_calls
    finally:
        _shim._import_sidebar_module = original_import  # type: ignore[assignment]


class TestSidekickEmbedding:
    """The Sidekick sidebar must be a third splitter pane, not a dock."""

    def test_no_floating_dock_at_startup(self, ui_setup_with_sidekick) -> None:
        ui_setup, _splitter, _calls = ui_setup_with_sidekick
        # The launcher main window itself is the only window we expect.
        top_level = [
            w for w in QApplication.topLevelWidgets() if w.isWindow() and w is not None
        ]
        # Filter to widgets that are descendants of QMainWindow — the
        # offscreen platform creates other helper top-levels (style proxy,
        # accessibility, etc.) we don't care about.  We only fail if any
        # QDockWidget appears as a top-level (i.e. floated).
        from PyQt6.QtWidgets import QDockWidget

        floating_docks = [
            w for w in top_level if isinstance(w, QDockWidget) and w.isFloating()
        ]
        assert not floating_docks, (
            f"unexpected floating QDockWidget(s): {floating_docks!r}"
        )

    def test_main_splitter_has_three_panes(self, ui_setup_with_sidekick) -> None:
        _ui, splitter, _calls = ui_setup_with_sidekick
        assert splitter.count() == 3, (
            f"expected 3 panes in the main splitter "
            f"(global sidebar | content | sidekick); got {splitter.count()}"
        )

    def test_sidekick_widget_is_child_of_main_splitter(
        self, ui_setup_with_sidekick
    ) -> None:
        _ui, splitter, _calls = ui_setup_with_sidekick
        # The Sidekick widget should be the rightmost (last) child.
        last = splitter.widget(splitter.count() - 1)
        assert last is not None
        # It must have come from our fake factory (objectName check).
        assert last.objectName() == "fakeSidekickSidebar", (
            "the rightmost splitter pane must be the embedded Sidekick "
            "widget produced by create_tools_sidebar()"
        )

    def test_install_does_not_call_adddockwidget(self, ui_setup_with_sidekick) -> None:
        ui_setup, _splitter, _calls = ui_setup_with_sidekick
        # The QMainWindow API ``addDockWidget`` should not have populated
        # any dock area with a Sidekick dock.  We assert no QDockWidget
        # objects with the unified-tools-sidebar object name exist.
        from PyQt6.QtWidgets import QDockWidget

        docks = ui_setup.findChildren(QDockWidget)
        sidekick_docks = [
            d
            for d in docks
            if d.objectName().lower().startswith("unifiedtools")
            or d.windowTitle() == "Tools"
        ]
        assert not sidekick_docks, (
            f"unexpected Sidekick dock widget(s) on a frameless main "
            f"window: {[d.objectName() for d in sidekick_docks]!r}"
        )
