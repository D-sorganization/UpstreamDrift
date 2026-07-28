"""Regression tests for launcher menu crashes (#8023) and console output (#8003).

Two independent defects are covered:

* ``View > Edit Layout Mode`` dereferenced ``self.btn_modify_layout``, an
  attribute that no UI setup path ever created. The resulting
  ``AttributeError`` inside a Qt slot aborts the process (``0xC0000409`` on
  Windows), so the whole app died when the menu item was clicked.
* Subprocess stdout was marshalled to the console with a bare
  ``QTimer.singleShot(0, ...)`` called from a plain ``threading.Thread``.
  ``singleShot`` posts to the *calling* thread's event loop, and reader
  threads have none, so the timer never fired.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.launchers import launcher_ui_setup, upstream_drift_launcher  # noqa: E402
from src.launchers.launcher_ui_setup import (  # noqa: E402
    ProcessOutputRelay,
    UISetupManager,
)

pytestmark = pytest.mark.unit


def _walk_menu(menu, out):
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None:
            _walk_menu(submenu, out)
        else:
            out.append(action)
    return out


class TestEditLayoutModeAction:
    """``View > Edit Layout Mode`` must not abort the process (#8023)."""

    def test_btn_modify_layout_is_not_dereferenced(self) -> None:
        """No launcher module may dereference the phantom attribute."""
        from src.launchers import settings_dialog

        for module in (upstream_drift_launcher, launcher_ui_setup, settings_dialog):
            source = module.__file__
            assert source is not None
            with open(source, encoding="utf-8") as handle:
                text = handle.read()
            assert ".btn_modify_layout" not in text, (
                f"{module.__name__} still dereferences btn_modify_layout, which is "
                "never assigned — see #8023"
            )

    @pytest.mark.slow
    def test_triggering_the_real_menu_action_survives(self, qapp_or_skip) -> None:
        """Build the real launcher and trigger the real QAction."""
        launcher = upstream_drift_launcher.UpstreamDriftLauncher()
        try:
            actions = _walk_menu(launcher.menu_bar, [])
            by_text = {a.text().replace("&", ""): a for a in actions}

            for label in ("Edit Layout Mode", "Context Help Panel"):
                action = by_text.get(label)
                assert action is not None, f"{label!r} action missing from View menu"
                action.trigger()
                assert action.isChecked() is True
                action.trigger()
                assert action.isChecked() is False
        finally:
            with_cleanup = getattr(launcher, "process_manager", None)
            if with_cleanup is not None:
                cleanup = getattr(with_cleanup, "cleanup_all", None)
                if callable(cleanup):
                    cleanup()
            launcher.deleteLater()


class _StubLauncher(QObject):
    """Minimal QObject launcher stand-in for the console relay."""


class TestProcessOutputRelay:
    """Subprocess output must reach the GUI thread (#8003)."""

    def test_no_bare_single_shot_in_on_process_output(self) -> None:
        """The broken cross-thread hand-off must not come back."""
        import inspect

        source = inspect.getsource(UISetupManager._on_process_output)
        assert "singleShot(" not in source, (
            "QTimer.singleShot cannot marshal from a non-Qt reader thread — #8003"
        )

    def test_relay_delivers_lines_emitted_from_a_worker_thread(
        self, qapp_or_skip
    ) -> None:
        """A line emitted off-thread is delivered on the GUI thread."""
        app = qapp_or_skip
        received: list[tuple[str, str, int]] = []
        gui_thread_id = threading.get_ident()

        parent = _StubLauncher()
        relay = ProcessOutputRelay(
            lambda name, line: received.append((name, line, threading.get_ident())),
            parent,
        )

        def worker() -> None:
            relay.line_received.emit("demo", "hello from the reader thread")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)

        QTimer.singleShot(200, app.quit)
        app.exec()

        assert received == [("demo", "hello from the reader thread", gui_thread_id)], (
            "line must be delivered exactly once, on the GUI thread"
        )

    def test_ui_setup_manager_emits_through_the_relay(self, qapp_or_skip) -> None:
        """``_on_process_output`` routes through the queued relay."""
        app = qapp_or_skip
        received: list[tuple[str, str]] = []

        manager = UISetupManager(_StubLauncher())
        object.__setattr__(
            manager,
            "_append_console_line",
            lambda name, line: received.append((name, line)),
        )
        manager._ensure_console_relay()

        thread = threading.Thread(
            target=lambda: manager._on_process_output("demo", "line from thread")
        )
        thread.start()
        thread.join(timeout=5)

        assert received == [], "delivery must be deferred, never synchronous"

        QTimer.singleShot(200, app.quit)
        app.exec()

        assert received == [("demo", "line from thread")]


@pytest.fixture
def qapp_or_skip():
    """Return a QApplication, creating one if the suite has not already."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
