"""Headless smoke test: standalone Sidekick shell surfaces its sidebar tabs.

Marker: headless_safe — requires ``QT_QPA_PLATFORM=offscreen``.

This complements ``test_window.py`` (which covers layout, profiles, and
accessors) by asserting the standalone shell genuinely reuses the existing
``UnifiedToolsSidebar`` widget and that the sidebar's published tab API is
reachable *without* the UpstreamDrift launcher, model registry, or physics
engines being imported.

Why a separate test
-------------------
The epic (#5969) requires that ``python -m sidekick`` opens "a focused window
with the AI chat and the tools sidebar, with no dependency on the UpstreamDrift
launcher process or model registry."  The most load-bearing acceptance check is
that the tools sidebar tabs are present on the standalone surface and are built
through the shared ``tools_sidebar`` package — not re-implemented.

When real PyQt6 is unavailable the project ``conftest`` installs a MagicMock
shim; in that case these tests skip cleanly (matching ``test_window.py``).
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

pytestmark = [pytest.mark.headless_safe]


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def app() -> Any:
    """Real-PyQt6 QApplication, or skip when only the mock shim is present."""
    try:
        import PyQt6

        # PYQT_VERSION_STR is a plain ``str`` on real PyQt6 and a MagicMock on
        # the conftest shim; this is the same gate used by test_window.py.
        if not isinstance(getattr(PyQt6, "PYQT_VERSION_STR", None), str):
            pytest.skip("Real PyQt6 not available (mock shim detected)")

        from PyQt6.QtWidgets import QApplication

        return QApplication.instance() or QApplication(sys.argv)
    except ImportError:
        pytest.skip("PyQt6 not available")


@pytest.fixture
def session_store(tmp_path: Any) -> Any:
    try:
        from sidekick.standalone.session_store import StandaloneSessionStore

        return StandaloneSessionStore(tmp_path)
    except ImportError:
        pytest.skip("session_store not available")


def _make_config(session_store: Any, profile: str = "chat-first") -> Any:
    from sidekick.standalone.window import StandaloneSidekickConfig

    return StandaloneSidekickConfig(
        profile=profile,
        theme_name=None,
        session_store=session_store,
    )


# ---------------------------------------------------------------------------
# The standalone shell reuses the real UnifiedToolsSidebar
# ---------------------------------------------------------------------------


class TestStandaloneSidebarReuse:
    def test_sidebar_is_unified_tools_sidebar(
        self, app: Any, session_store: Any
    ) -> None:
        """The shell must reuse the existing widget, not a placeholder."""
        from sidekick.standalone.window import StandaloneSidekickWindow
        from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

        win = StandaloneSidekickWindow(_make_config(session_store, "chat-first"))
        try:
            sidebar = win.sidebar()
            assert isinstance(sidebar, UnifiedToolsSidebar), (
                "standalone shell must host the real UnifiedToolsSidebar, "
                f"got {type(sidebar).__name__!r} (placeholder fallback?)"
            )
        finally:
            win.close()

    def test_sidebar_tab_api_reachable(self, app: Any, session_store: Any) -> None:
        """Published tab API is reachable on the standalone surface (LOD)."""
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store))
        try:
            sidebar = win.sidebar()
            available = sidebar.available_tab_ids()
            visible = sidebar.visible_tab_ids()
            assert isinstance(available, list)
            assert isinstance(visible, list)
            # Every visible tab must be a known/available tab.
            assert set(visible).issubset(set(available)), (
                f"visible tabs {visible} not a subset of available {available}"
            )
        finally:
            win.close()


# ---------------------------------------------------------------------------
# Expected default tabs are present when the sidebar is given the shared
# default tab definitions (the canonical host pattern).
# ---------------------------------------------------------------------------


class TestExpectedTabsPresent:
    def test_explicit_default_tabs_surface(self, app: Any) -> None:
        """Configuring the sidebar with definitions surfaces those tab ids.

        This asserts the *reuse contract*: the standalone shell does not need
        to re-implement tab construction — handing the shared
        ``UnifiedToolsSidebar`` a set of ``SidebarTabDefinition`` objects yields
        exactly those tab ids back through the published API.
        """
        from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar
        from sidekick.ui.tools_sidebar.tab_definition import SidebarTabDefinition

        def _factory(_sidebar: Any) -> Any:
            from PyQt6.QtWidgets import QLabel

            return QLabel("tab")

        expected = ("calculator", "workspace", "notes", "jupyter", "chat")
        definitions = [
            SidebarTabDefinition(
                tab_id=tab_id,
                title=tab_id.capitalize(),
                factory=_factory,
            )
            for tab_id in expected
        ]

        sidebar = UnifiedToolsSidebar(tab_definitions=definitions)
        try:
            available = sidebar.available_tab_ids()
            for tab_id in expected:
                assert tab_id in available, (
                    f"expected tab {tab_id!r} missing from {available}"
                )
        finally:
            sidebar.deleteLater()


# ---------------------------------------------------------------------------
# Hygiene: the standalone surface pulls in no launcher / model / physics deps.
# ---------------------------------------------------------------------------


class TestNoLauncherOrModelImports:
    def test_no_launcher_model_or_physics_imports(self) -> None:
        """Importing the standalone window must not drag in heavy host deps."""
        for key in list(sys.modules):
            if "sidekick.standalone.window" in key:
                del sys.modules[key]

        pre = set(sys.modules)
        try:
            import sidekick.standalone.window  # noqa: F401
        except ImportError:
            pytest.skip("standalone window not importable in this environment")

        new = set(sys.modules) - pre
        forbidden = ("src.launchers", "src.physics", "model_registry")
        bad = [m for m in new if any(tok in m for tok in forbidden)]
        assert not bad, f"standalone window imported forbidden deps: {bad}"
