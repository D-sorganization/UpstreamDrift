"""The OpenSim backend must never use the real display (NV-5, #11678)."""

from __future__ import annotations

import pytest

from src.tools.native_viewer_export.backends.opensim_simbody import (
    OpenSimSimbodyBackend,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_worker_command_goes_through_xvfb_run_with_its_own_screen() -> None:
    cmd = OpenSimSimbodyBackend().command()
    assert cmd[0] == "xvfb-run" and "-a" in cmd
    assert any("-screen 0" in part for part in cmd)
    assert cmd[-1].endswith("opensim_worker")


@pytest.mark.parametrize("display", ["", ":0", ":0.0"])
def test_worker_refuses_the_real_display(
    monkeypatch: pytest.MonkeyPatch, display: str
) -> None:
    pytest.importorskip("cv2")
    from src.tools.native_viewer_export.backends.opensim_worker import (
        require_virtual_display,
    )

    monkeypatch.setenv("NATIVE_VIEWER_XVFB", "1")
    monkeypatch.setenv("DISPLAY", display)
    with pytest.raises(RuntimeError, match="real display"):
        require_virtual_display()


def test_worker_refuses_to_run_without_the_xvfb_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tools.native_viewer_export.backends.opensim_worker import (
        require_virtual_display,
    )

    monkeypatch.delenv("NATIVE_VIEWER_XVFB", raising=False)
    monkeypatch.setenv("DISPLAY", ":99")
    with pytest.raises(RuntimeError, match="xvfb launcher"):
        require_virtual_display()


def test_worker_accepts_a_virtual_display(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tools.native_viewer_export.backends.opensim_worker import (
        require_virtual_display,
    )

    monkeypatch.setenv("NATIVE_VIEWER_XVFB", "1")
    monkeypatch.setenv("DISPLAY", ":99")
    assert require_virtual_display() == ":99"


def test_grab_window_raises_when_xwininfo_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tools.native_viewer_export.backends import opensim_worker

    def fake_run(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file or directory: 'xwininfo'")

    monkeypatch.setattr(opensim_worker.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="xwininfo is not installed"):
        opensim_worker.grab_window()


def test_grab_window_raises_when_xwd_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from src.tools.native_viewer_export.backends import opensim_worker

    def fake_run(cmd: list[str], *_args: object, **_kwargs: object) -> SimpleNamespace:
        if cmd[0] == "xwininfo":
            return SimpleNamespace(
                stdout='0x123456 (has no name): ()  "OpenSim Visualizer"\n'
            )
        if cmd[0] == "xwd":
            raise FileNotFoundError(2, "No such file or directory: 'xwd'")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(opensim_worker.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="xwd is not installed"):
        opensim_worker.grab_window()
