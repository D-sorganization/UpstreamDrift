"""Shared fixtures for launcher unit tests.

Provides a session-scoped ``qapp`` fixture (since pytest-qt is not a
declared dependency) and the ``ui_setup`` harness used by the #5624
regression suite (layout hierarchy, sidekick embedding, sidebar icons).
"""

from __future__ import annotations

import os

import pytest

# Force Qt to use the offscreen platform so headless CI / sandboxed
# runners do not fail to instantiate widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped ``QApplication`` instance.

    Mirrors ``tests/launchers/conftest.py::qapp``.  We define a local
    copy here because the launcher-tests directory's conftest is not
    visible to ``tests/unit/launcher/`` collection.
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def ui_setup(qapp):
    """Construct just enough of the launcher to call ``init_ui``.

    The full ``UpstreamDriftLauncher`` pulls in async startup workers, Docker
    probes, and a worker thread pool that we do not need here.  We
    mount the UI-setup mixin on a bare ``QMainWindow`` subclass that
    provides the handful of attribute/method stubs ``init_ui`` reads
    from its co-mixins.

    The ``_install_sidekick_sidebar`` method is borrowed from the real
    ``UpstreamDriftLauncher`` so the same code path #5624 fixes is exercised.
    """
    from PyQt6.QtWidgets import QApplication, QMainWindow

    from src.launchers.upstream_drift_launcher import (
        UpstreamDriftLauncher as _RealLauncher,
    )
    from src.launchers.launcher_ui_setup import LauncherUISetupMixin

    class _DummyLayoutManager:
        tile_scale = 0.75

        def set_view_mode(self, *_a, **_k) -> None: ...
        def set_tile_scale(self, *_a, **_k) -> None: ...
        def rebuild_grid(self, *_a, **_k) -> None: ...

    class _UIHarness(LauncherUISetupMixin, QMainWindow):  # type: ignore[misc]
        # Re-use the production install method so #5624 tests exercise
        # the real code path.
        _install_sidekick_sidebar = _RealLauncher._install_sidekick_sidebar
        _apply_sidekick_splitter_sizes = _RealLauncher._apply_sidekick_splitter_sizes

        def __init__(self) -> None:
            super().__init__()
            self.layout_manager = _DummyLayoutManager()
            self.docker_available = False

        # Safe no-op stubs for co-mixin handlers that init_ui wires up.
        def _show_preferences(self, *_a, **_k) -> None: ...
        def _toggle_layout_mode_from_menu(self, *_a, **_k) -> None: ...
        def _toggle_context_help(self, *_a, **_k) -> None: ...
        def _show_help_dialog(self, *_a, **_k) -> None: ...
        def _show_shortcuts_overlay(self, *_a, **_k) -> None: ...
        def _show_about_dialog(self, *_a, **_k) -> None: ...
        def _open_project_map(self, *_a, **_k) -> None: ...
        def _open_settings(self, *_a, **_k) -> None: ...
        def _show_help_dialog_topic(self, *_a, **_k) -> None: ...
        def _on_docker_mode_changed(self, *_a, **_k) -> None: ...
        def _on_wsl_mode_changed(self, *_a, **_k) -> None: ...
        def toggle_layout_mode(self, *_a, **_k) -> None: ...
        def open_layout_manager(self, *_a, **_k) -> None: ...
        def update_search_filter(self, *_a, **_k) -> None: ...
        def launch_simulation(self, *_a, **_k) -> None: ...
        def _setup_theme_menu(self, *_a, **_k) -> None: ...
        def apply_styles(self) -> None: ...

    window = _UIHarness()
    window.init_ui()
    yield window
    window.deleteLater()
    QApplication.processEvents()
