"""Tests for the shared window-icon / taskbar-identity helpers (synced from Tools).

Guards the favicon regression that recurred because earlier fixes only adjusted
the icon file path and asserted ``windowIcon() is not None`` — true even when the
Windows *taskbar* icon is wrong. The key assertion is that the AppUserModelID is
actually set (the missing piece) and that the icon is applied to both the
application and the window.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PyQt6")

from src.shared.python.ui import (  # noqa: E402
    apply_window_icon,
    resolve_icon_path,
    set_app_user_model_id,
)

pytestmark = pytest.mark.unit


class _FakeIconTarget:
    """Records ``setWindowIcon`` calls without needing a real QWidget."""

    def __init__(self) -> None:
        self.icons: list[Any] = []

    def setWindowIcon(self, icon: Any) -> None:  # noqa: N802 - Qt API name
        self.icons.append(icon)


def test_resolve_icon_path_returns_first_existing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.ico"
    present = tmp_path / "present.png"
    present.write_bytes(b"icon")
    assert resolve_icon_path([missing, present]) == present


def test_resolve_icon_path_none_when_no_candidate_exists(tmp_path: Path) -> None:
    assert resolve_icon_path([tmp_path / "nope.ico"]) is None


def test_set_app_user_model_id_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        set_app_user_model_id("   ")


def test_set_app_user_model_id_noop_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert set_app_user_model_id("D-sorganization.UpstreamDrift") is False


def test_set_app_user_model_id_calls_windows_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The taskbar fix: SetCurrentProcessExplicitAppUserModelID is invoked with
    the exact id. This is the assertion the previous favicon fix lacked."""
    monkeypatch.setattr(sys, "platform", "win32")
    recorded: list[str] = []
    fake_shell32 = types.SimpleNamespace(
        SetCurrentProcessExplicitAppUserModelID=lambda app_id: recorded.append(app_id)
    )
    fake_ctypes = types.SimpleNamespace(
        windll=types.SimpleNamespace(shell32=fake_shell32)
    )
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    assert set_app_user_model_id("D-sorganization.UpstreamDrift") is True
    assert recorded == ["D-sorganization.UpstreamDrift"]


def test_apply_window_icon_sets_app_and_window(tmp_path: Path) -> None:
    icon_file = tmp_path / "app.ico"
    icon_file.write_bytes(b"ico")
    app = _FakeIconTarget()
    window = _FakeIconTarget()

    result = apply_window_icon(
        app=app,
        window=window,
        icon_candidates=[icon_file],
        icon_factory=lambda path: ("ICON", path),
    )

    assert result == icon_file
    assert app.icons == [("ICON", str(icon_file))]
    assert window.icons == [("ICON", str(icon_file))]


def test_apply_window_icon_declares_app_user_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    icon_file = tmp_path / "app.ico"
    icon_file.write_bytes(b"ico")
    called: list[str] = []
    monkeypatch.setattr(
        "src.shared.python.ui.window_icon.set_app_user_model_id",
        lambda app_id: called.append(app_id) or True,
    )

    apply_window_icon(
        app=_FakeIconTarget(),
        window=_FakeIconTarget(),
        icon_candidates=[icon_file],
        app_id="D-sorganization.UpstreamDrift",
        icon_factory=lambda path: path,
    )

    assert called == ["D-sorganization.UpstreamDrift"]


def test_apply_window_icon_missing_icon_returns_none(tmp_path: Path) -> None:
    window = _FakeIconTarget()
    result = apply_window_icon(
        app=_FakeIconTarget(),
        window=window,
        icon_candidates=[tmp_path / "nope.ico"],
        icon_factory=lambda path: path,
    )
    assert result is None
    assert window.icons == []


def test_apply_window_icon_requires_window() -> None:
    with pytest.raises(TypeError):
        apply_window_icon(app=None, window=None, icon_candidates=[])
