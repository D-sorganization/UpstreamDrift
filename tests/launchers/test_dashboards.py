"""Tests for various dashboard launchers."""

import importlib
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.dashboard import launcher as launcher_module


@pytest.mark.unit
def test_mujoco_dashboard_main() -> None:
    pytest.importorskip("mujoco")
    with patch("src.launchers.mujoco_dashboard.launch_dashboard") as mock_launch:
        # Resolve ``main`` from the module *inside* the patch context. Binding it
        # at import time is unsafe: other tests evict launcher modules from
        # ``sys.modules``, after which a module-level ``main`` refers to the old
        # module dict while ``patch`` targets the freshly imported one. The patch
        # then misses and the real ``launch_dashboard`` blocks in a Qt event loop
        # until CI times out (issue #9183).
        importlib.import_module("src.launchers.mujoco_dashboard").main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "MuJoCoPhysicsEngine"
        assert kwargs["title"] == "MuJoCo Golf Analysis Dashboard (Unified)"


@pytest.mark.unit
def test_pinocchio_dashboard_main() -> None:
    pytest.importorskip("pinocchio")
    with patch("src.launchers.pinocchio_dashboard.launch_dashboard") as mock_launch:
        # See test_mujoco_dashboard_main for why ``main`` is resolved here.
        importlib.import_module("src.launchers.pinocchio_dashboard").main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "PinocchioPhysicsEngine"
        assert kwargs["title"] == "Pinocchio Golf Analysis Dashboard"


@pytest.mark.unit
def test_default_event_loop_runner_refuses_to_block_under_pytest() -> None:
    """The event-loop seam must fail fast rather than hang a test worker.

    Regression guard for issue #9183: when a launcher reaches the real event
    loop during a test run there is nothing to post a quit event, so the worker
    blocks until the suite timeout. Raising immediately keeps the failure
    diagnosable and cheap.
    """
    qt_app = MagicMock()

    with pytest.raises(RuntimeError, match="blocking Qt event loop"):
        launcher_module._default_event_loop_runner(qt_app)

    qt_app.exec.assert_not_called()


@pytest.mark.unit
def test_default_event_loop_runner_runs_loop_outside_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a test session the seam still runs the real event loop."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    qt_app = MagicMock()
    qt_app.exec.return_value = 0

    assert launcher_module._default_event_loop_runner(qt_app) == 0
    qt_app.exec.assert_called_once_with()
