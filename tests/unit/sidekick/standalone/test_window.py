"""Tests for StandaloneSidekickWindow — T2 (#5980).

Marker: headless_safe — uses QT_QPA_PLATFORM=offscreen.

Covers:
  - invalid profile raises ValueError at config construction
  - window constructs for chat-first and calc-first
  - window title is "Sidekick"
  - splitter ratios match documented 60/40 split within ±5%
  - panel_for / sidebar public accessors
  - window module does not import src.launchers.*
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
def session_store(tmp_path: Any) -> Any:
    try:
        from sidekick.standalone.session_store import StandaloneSessionStore

        return StandaloneSessionStore(tmp_path)
    except ImportError:
        pytest.skip("session_store not available")


@pytest.fixture
def app() -> Any:
    try:
        import PyQt6

        # When real PyQt6 is unavailable the project conftest installs a
        # MagicMock shim.  PYQT_VERSION_STR is a plain str on real PyQt6;
        # a MagicMock on the shim.
        if not isinstance(getattr(PyQt6, "PYQT_VERSION_STR", None), str):
            pytest.skip("Real PyQt6 not available (mock shim detected)")

        from PyQt6.QtWidgets import QApplication

        return QApplication.instance() or QApplication(sys.argv)
    except ImportError:
        pytest.skip("PyQt6 not available")


def _make_config(session_store: Any, profile: str = "chat-first") -> Any:
    from sidekick.standalone.window import StandaloneSidekickConfig

    return StandaloneSidekickConfig(
        profile=profile,
        theme_name=None,
        session_store=session_store,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_invalid_profile_raises_value_error(self, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickConfig

        with pytest.raises(ValueError, match="invalid-profile"):
            StandaloneSidekickConfig(
                profile="invalid-profile",
                theme_name=None,
                session_store=session_store,
            )

    def test_valid_chat_first(self, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickConfig

        cfg = StandaloneSidekickConfig(
            profile="chat-first", theme_name=None, session_store=session_store
        )
        assert cfg.profile == "chat-first"

    def test_valid_calc_first(self, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickConfig

        cfg = StandaloneSidekickConfig(
            profile="calc-first", theme_name=None, session_store=session_store
        )
        assert cfg.profile == "calc-first"


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


class TestWindowConstruction:
    def test_chat_first_constructs(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store, "chat-first"))
        assert win is not None
        win.close()

    def test_calc_first_constructs(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store, "calc-first"))
        assert win is not None
        win.close()

    def test_window_title_is_sidekick(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store))
        assert win.windowTitle() == "Sidekick"
        win.close()


# ---------------------------------------------------------------------------
# Splitter layout ratios
# ---------------------------------------------------------------------------


class TestSplitterRatio:
    def test_chat_first_ratio(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store, "chat-first"))
        win.resize(1280, 800)
        win.show()
        sizes = win.splitter_handle_positions()
        assert len(sizes) == 2
        total = sum(sizes)
        assert total > 0
        chat_ratio = sizes[0] / total
        assert (
            abs(chat_ratio - 0.6) < 0.05
        ), f"chat-first: expected ~0.60 left ratio, got {chat_ratio:.3f}"
        win.close()

    def test_calc_first_ratio(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store, "calc-first"))
        win.resize(1280, 800)
        win.show()
        sizes = win.splitter_handle_positions()
        assert len(sizes) == 2
        total = sum(sizes)
        assert total > 0
        sidebar_ratio = sizes[0] / total
        assert (
            abs(sidebar_ratio - 0.6) < 0.05
        ), f"calc-first: expected ~0.60 left ratio, got {sidebar_ratio:.3f}"
        win.close()


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


class TestPublicAccessors:
    def test_panel_for_chat_first(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store, "chat-first"))
        panel = win.panel_for("chat-first")
        assert panel is not None
        win.close()

    def test_panel_for_calc_first(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store, "calc-first"))
        panel = win.panel_for("calc-first")
        assert panel is not None
        win.close()

    def test_sidebar_accessor(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store))
        sidebar = win.sidebar()
        assert sidebar is not None
        win.close()

    def test_panel_for_unknown_raises(self, app: Any, session_store: Any) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store))
        with pytest.raises(ValueError):
            win.panel_for("unknown")
        win.close()


# ---------------------------------------------------------------------------
# Hygiene: no src.launchers.* imports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Profile switching
# ---------------------------------------------------------------------------


class TestProfileSwitch:
    def test_switch_profile_reorders_and_reflows(
        self, app: Any, session_store: Any
    ) -> None:
        from sidekick.standalone.window import StandaloneSidekickWindow

        win = StandaloneSidekickWindow(_make_config(session_store, "chat-first"))
        win.resize(1280, 800)
        win.show()

        # Check initial state
        sizes_chat_first = win.splitter_handle_positions()
        chat_ratio = sizes_chat_first[0] / sum(sizes_chat_first)
        assert abs(chat_ratio - 0.6) < 0.05

        # Access internal layout to check widget order without breaking encapsulation too badly
        assert win._splitter.widget(0) is win._chat_panel
        assert win._splitter.widget(1) is win._sidebar_panel

        # Switch to calc-first
        win._switch_profile("calc-first")

        # Check new state
        sizes_calc_first = win.splitter_handle_positions()
        calc_ratio = sizes_calc_first[0] / sum(sizes_calc_first)
        assert abs(calc_ratio - 0.6) < 0.05
        assert win._splitter.widget(0) is win._sidebar_panel
        assert win._splitter.widget(1) is win._chat_panel

        win.close()


class TestNoLaunchersImport:
    def test_window_module_does_not_import_src_launchers(self) -> None:
        for key in list(sys.modules):
            if "sidekick.standalone.window" in key:
                del sys.modules[key]

        pre = set(sys.modules)
        try:
            import sidekick.standalone.window  # noqa: F401
        except ImportError:
            pytest.skip("standalone window not importable in this environment")

        new = set(sys.modules) - pre
        bad = [m for m in new if "src.launchers" in m]
        assert not bad, f"Unexpected src.launchers imports: {bad}"
