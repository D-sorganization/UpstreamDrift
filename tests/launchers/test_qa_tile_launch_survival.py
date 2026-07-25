"""The launcher must survive any tile launch failure (epic #8062).

These are the high-value shape from the functional-QA campaign: drive a real
tile launch with its dependency mocked absent and assert the *host* is still
standing and told the user something specific.

Issues covered: #8065, #8066, #8068, #8069, #8070, #8072, #8087.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QLabel, QMainWindow

from src.launchers.launcher_simulation import SimulationManager

WIN_DLL_INIT_ERROR = (
    "[WinError 1114] A dynamic link library (DLL) initialization routine failed"
)


class _Model:
    def __init__(self, model_id: str, name: str, model_type: str) -> None:
        self.id = model_id
        self.name = name
        self.type = model_type
        self.path = "src/config/models.yaml"
        self.embed_adapter = None
        self.engine_type = "mujoco"
        self.source_root = None
        self.provider = None


class _Host(QMainWindow):
    """Minimal stand-in for UpstreamDriftLauncher's delegation behaviour."""

    def __getattr__(self, name: str) -> Any:
        manager = self.__dict__.get("manager")
        if manager is not None and (
            name in manager.__dict__ or hasattr(type(manager), name)
        ):
            attr = getattr(manager, name)
            import types

            if isinstance(attr, types.MethodType):
                return types.MethodType(attr.__func__, self)
            return attr
        raise AttributeError(name)

    def __init__(self) -> None:
        super().__init__()
        self.manager = SimulationManager(self)
        self.show_toast = MagicMock()
        self.lbl_status = MagicMock()
        self.console_lines: list[str] = []
        self.process_manager = MagicMock()
        self.model_handler_registry = MagicMock()
        self.docker_launcher = MagicMock()
        self.running_processes: dict[str, Any] = {}
        self.docker_available = False
        for name in ("chk_docker", "chk_wsl", "chk_gpu"):
            checkbox = MagicMock()
            checkbox.isChecked.return_value = False
            setattr(self, name, checkbox)
        self.models = {"tile": _Model("tile", "Pose Studio", "special_app")}
        self.selected_model = "tile"

    def _append_console_line(self, source: str, text: str) -> None:
        self.console_lines.append(f"{source}: {text}")

    def _get_model(self, model_id: str) -> Any:
        return self.models.get(model_id)


@pytest.fixture
def host(qapp: Any) -> _Host:  # noqa: ARG001 - qapp ensures a QApplication exists
    widget = _Host()
    yield widget
    widget.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("exc", "expected_fragment"),
    [
        (
            ModuleNotFoundError("No module named 'sidekick.lab.bio._c3d_marker_set'"),
            "sidekick.lab.bio._c3d_marker_set",
        ),
        (OSError(WIN_DLL_INIT_ERROR), "1114"),
        (
            FileNotFoundError("[WinError 2] The system cannot find the file specified"),
            "MATLAB",
        ),
        (AttributeError("'MainWidget' object has no attribute 'act_undo'"), "console"),
    ],
    ids=["missing-gui-extra", "broken-mujoco-dll", "missing-matlab", "unexpected-bug"],
)
def test_launcher_survives_any_tile_launch_failure(
    host: _Host, exc: BaseException, expected_fragment: str
) -> None:
    """Every launch failure is contained and explained; nothing propagates out
    of the Qt slot, so the global excepthook never fires (#8066, #8070, #8072)."""
    with (
        patch.object(host.manager, "_check_local_dependencies", return_value=True),
        patch.object(host.manager, "_try_launch_docker", return_value=False),
        patch.object(host.manager, "_execute_local_launch", side_effect=exc),
        patch.object(SimulationManager, "_open_launch_failure_dialog") as warning,
    ):
        host.manager.launch_simulation()

    assert warning.called, "user was given no in-product error"
    message = warning.call_args.args[0]
    assert "Pose Studio" in message
    assert expected_fragment in message
    assert "Traceback" not in message

    host.show_toast.assert_called()
    host.lbl_status.setText.assert_called_with("> Ready")
    assert host.isVisible() is False or host.isWidgetType()


@pytest.mark.unit
def test_provider_tile_without_runtime_shows_install_command(host: _Host) -> None:
    """A provider tile whose engine runtime is missing must produce an
    actionable message rather than a silent no-op (#8087)."""
    from src.launchers import launcher_provider_asset_handler as mod

    error = mod.EngineRuntimeUnavailableError(
        "Drake Squat needs the Drake runtime, which is not installed.\n\n"
        "Install it with:\n    pip install drake\n\n"
        "The launcher is still running - other tiles are unaffected."
    )

    with (
        patch.object(host.manager, "_check_local_dependencies", return_value=True),
        patch.object(host.manager, "_try_launch_docker", return_value=False),
        patch.object(host.manager, "_execute_local_launch", side_effect=error),
        patch.object(SimulationManager, "_open_launch_failure_dialog") as warning,
    ):
        host.manager.launch_simulation()

    message = warning.call_args.args[0]
    assert message == str(error)
    assert "pip install drake" in message


@pytest.mark.unit
def test_child_that_exits_immediately_is_reported(host: _Host) -> None:
    """ "Launched <tile> (PID: n)" is not proof the tool came up (#8065, #8069)."""
    dead_child = MagicMock()
    dead_child.poll.return_value = 1

    captured: list[Any] = []
    with patch(
        "src.launchers.launcher_failure_reporting.QTimer.singleShot",
        side_effect=lambda _ms, fn: captured.append(fn),
    ):
        host.manager._watch_child_process("Shot Tracer", dead_child)

    assert captured, "no liveness check was scheduled"
    captured[0]()

    host.show_toast.assert_called_with("Shot Tracer exited immediately", "error")
    assert any("gui-tools" in line for line in host.console_lines)


@pytest.mark.unit
def test_live_child_is_not_reported_as_failed(host: _Host) -> None:
    live_child = MagicMock()
    live_child.poll.return_value = None

    captured: list[Any] = []
    with patch(
        "src.launchers.launcher_failure_reporting.QTimer.singleShot",
        side_effect=lambda _ms, fn: captured.append(fn),
    ):
        host.manager._watch_child_process("Shot Tracer", live_child)
    captured[0]()

    host.show_toast.assert_not_called()


@pytest.mark.unit
def test_gait_dashboard_shows_actionable_panel_not_a_blank_area(
    qapp: Any,  # noqa: ARG001 - ensures a QApplication exists
) -> None:
    """A MuJoCo DLL failure left the Gait window blank with a raw WinError
    string and no recovery guidance (#8068)."""
    from src.launchers import exercise_dashboard as mod

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(WIN_DLL_INIT_ERROR)

    with patch("src.launchers.mujoco_dashboard.MuJoCoDashboard", side_effect=_boom):
        dashboard = mod.ExerciseDashboard("gait")

    panel = dashboard._current_widget
    assert isinstance(panel, QLabel)
    text = panel.text()
    assert "MuJoCo" in text
    assert "1114" in text
    assert "Visual C++" in text
    assert "Other engines you can select" in text
    dashboard.close()
