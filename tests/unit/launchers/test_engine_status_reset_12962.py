"""Issue #12962 (refs #12678): closing an engine tab resets its Running status."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QLabel, QWidget

from src.launchers.engine_status import (
    engine_running_text,
    is_engine_running_status,
    reset_status_if_engine_closed,
)
from src.shared.python.gui_pkg.draggable_tabs import DraggableTabWidget


@pytest.fixture(autouse=True)
def _no_close_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let close_tab raise a modal confirmation dialog."""
    from types import SimpleNamespace

    from src.shared.python.ui import preferences_dialog

    monkeypatch.setattr(
        preferences_dialog.UserPreferences,
        "load",
        classmethod(lambda cls: SimpleNamespace(confirm_close_tabs="never")),
    )


@pytest.mark.unit
def test_running_text_roundtrips_through_matcher() -> None:
    assert engine_running_text("MuJoCo") == "* MuJoCo Running"
    assert engine_running_text("MuJoCo", "WSL") == "* MuJoCo Running (WSL)"
    for text in ("* MuJoCo Running", "> MuJoCo Running", "* MuJoCo Running (Docker)"):
        assert is_engine_running_status(text, "MuJoCo")


@pytest.mark.unit
def test_matcher_rejects_other_engines_and_idle_text() -> None:
    assert not is_engine_running_status("* Drake Running", "MuJoCo")
    assert not is_engine_running_status("* MuJoCo Runningfoo", "MuJoCo")
    assert not is_engine_running_status("System Ready", "MuJoCo")
    assert not is_engine_running_status("* MuJoCo Running", "")


@pytest.mark.unit
def test_helpers_validate_arguments(qapp) -> None:
    with pytest.raises(ValueError):
        engine_running_text("  ")
    with pytest.raises(TypeError):
        engine_running_text(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        is_engine_running_status(None, "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        reset_status_if_engine_closed(None, "x", lambda: None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        reset_status_if_engine_closed(QLabel("x"), "x", None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_reset_only_when_closed_engine_is_the_running_one(qapp) -> None:
    label = QLabel("* MuJoCo Running")
    restore = MagicMock()
    assert reset_status_if_engine_closed(label, "Drake", restore) is False
    restore.assert_not_called()
    assert reset_status_if_engine_closed(label, "MuJoCo", restore) is True
    restore.assert_called_once_with()


@pytest.mark.unit
@pytest.mark.parametrize("eligible", [False, True])
def test_tab_widget_emits_tab_closed_for_destroy_and_background(
    qapp, monkeypatch: pytest.MonkeyPatch, eligible: bool
) -> None:
    tabs = DraggableTabWidget(core_tabs={"Home"})
    tabs.addTab(QWidget(), "Home")
    widget = QWidget()
    if eligible:
        tabs.add_background_tab(widget, "MuJoCo")
    else:
        tabs.addTab(widget, "MuJoCo")
    closed: list[str] = []
    tabs.tab_closed.connect(closed.append)
    tabs.close_tab(1)
    assert closed == ["MuJoCo"]
    assert tabs.count() == 1


@pytest.mark.unit
def test_tab_close_resets_launcher_status_label(qapp) -> None:
    tabs = DraggableTabWidget(core_tabs={"Home"})
    tabs.addTab(QWidget(), "Home")
    tabs.addTab(QWidget(), "MuJoCo")
    label = QLabel(engine_running_text("MuJoCo"))
    tabs.tab_closed.connect(
        lambda title: reset_status_if_engine_closed(
            label, title, lambda: label.setText("System Ready")
        )
    )
    tabs.close_tab(1)
    assert label.text() == "System Ready"
