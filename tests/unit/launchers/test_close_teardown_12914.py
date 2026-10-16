"""Regression for UD#12914: closing the window after a MuJoCo session crashed.

Embedded tool cleanup was only wired to ``QWidget.destroyed`` (fires after the
C++ object and its GL context are gone), and the sim widget never released its
``mujoco.Renderer`` explicitly, so the renderer was freed after the GL
context during interpreter teardown (access violation, 0xC0000005).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]
MUJOCO_PY = REPO / "src/engines/physics_engines/mujoco/python"


@pytest.mark.unit
def test_cleanup_embedded_tools_calls_every_registered_tool_once() -> None:
    from src.launchers import launcher_shutdown

    good, bad = mock.Mock(), mock.Mock()
    bad.cleanup.side_effect = RuntimeError("boom")
    registry = {"a": bad, "b": good}
    with mock.patch.object(launcher_shutdown, "EMBEDDABLE_TOOL_REGISTRY", registry):
        launcher_shutdown.cleanup_embedded_tools()
    bad.cleanup.assert_called_once_with()
    good.cleanup.assert_called_once_with()  # a raising tool must not stop others


@pytest.mark.unit
def test_main_widget_cleanup_releases_sim_widget_renderer() -> None:
    pytest.importorskip("PyQt6")
    pytest.importorskip("mujoco")
    sys.path.insert(0, str(MUJOCO_PY))
    from mujoco_humanoid_golf.gui.core.main_widget import MainWidget

    sim = mock.Mock()
    fake = SimpleNamespace(_inner=SimpleNamespace(status_timer=None, sim_widget=sim))
    MainWidget.cleanup(fake)  # type: ignore[arg-type]
    sim.stop_simulation.assert_called_once_with()
    sim.release_resources.assert_called_once_with()


_SCRIPT = textwrap.dedent(
    """
    import os, sys
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path[:0] = [{mj!r}, {repo!r}]
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication([])
    from mujoco_humanoid_golf.sim_widget import MuJoCoSimWidget
    from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML
    w = MuJoCoSimWidget(width=160, height=120, fps=30)
    w.load_model_from_xml(DOUBLE_PENDULUM_XML)
    w.show()
    for _ in range(10):
        app.processEvents()
    assert w.renderer is not None
    w.release_resources()
    assert w.renderer is None
    w.release_resources()  # idempotent
    w.close()
    print("OK")
    """
)


@pytest.mark.unit
def test_sim_widget_release_resources_then_exit_clean() -> None:
    pytest.importorskip("PyQt6")
    pytest.importorskip("mujoco")
    proc = subprocess.run(  # noqa: S603 - fixed argv, test-only
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-c",
            _SCRIPT.format(mj=str(MUJOCO_PY), repo=str(REPO)),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout
    assert "Windows fatal exception" not in proc.stderr
