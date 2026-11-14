"""Regression tests for LauncherService.get_running_processes() (#14227)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.services.launcher_service import LauncherService

pytestmark = pytest.mark.unit


class _LegacyProcessManager:
    """Process manager double exposing only ``running_processes``."""

    def __init__(self, running_processes: dict[str, MagicMock]) -> None:
        self.running_processes = running_processes


class _ManagedProcessManager:
    """Process manager double exposing ``list_processes()`` and ``running_processes``."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        running_processes: dict[str, MagicMock],
    ) -> None:
        self._rows = rows
        self.running_processes = running_processes

    def list_processes(self) -> list[dict[str, object]]:
        return list(self._rows)


def test_get_running_processes_ignores_auto_mocked_list_processes(
    tmp_path: Path,
) -> None:
    """Un-spec'd MagicMock ProcessManager must still expose running_processes entries."""
    service = LauncherService(repo_root=tmp_path)
    running_proc = MagicMock(pid=12345)
    running_proc.poll.return_value = None
    exited_proc = MagicMock(pid=99999)
    exited_proc.poll.return_value = 1

    service._process_manager = MagicMock(
        running_processes={
            "MuJoCo Humanoid Golf": running_proc,
            "Drake Golf Model": exited_proc,
        }
    )

    processes = service.get_running_processes()

    assert processes == {
        "MuJoCo Humanoid Golf": {"pid": 12345, "running": True, "exit_code": None},
        "Drake Golf Model": {"pid": 99999, "running": False, "exit_code": 1},
    }
    running_proc.poll.assert_called_once()
    exited_proc.poll.assert_called_once()


def test_get_running_processes_prefers_list_processes_without_double_polling(
    tmp_path: Path,
) -> None:
    """Managed list_processes() rows preserve full metadata and are not re-polled."""
    service = LauncherService(repo_root=tmp_path)
    managed_proc = MagicMock(pid=4242)
    managed_proc.poll.return_value = None
    extra_proc = MagicMock(pid=5252)
    extra_proc.poll.return_value = 0

    managed_row: dict[str, object] = {
        "id": "proc-1",
        "key": "mujoco_unified",
        "slot_name": "MuJoCo",
        "kind": "local",
        "state": "running",
        "pid": 4242,
        "running": True,
        "exit_code": None,
        "last_stderr_tail": "",
    }
    service._process_manager = _ManagedProcessManager(
        rows=[managed_row],
        running_processes={"MuJoCo": managed_proc, "External": extra_proc},
    )

    processes = service.get_running_processes()

    assert processes == {
        "MuJoCo": managed_row,
        "External": {"pid": 5252, "running": False, "exit_code": 0},
    }
    managed_proc.poll.assert_not_called()
    extra_proc.poll.assert_called_once()


def test_get_running_processes_legacy_manager_fallback(tmp_path: Path) -> None:
    """Process managers without list_processes() fall back to running_processes."""
    service = LauncherService(repo_root=tmp_path)
    proc = MagicMock(pid=777)
    proc.poll.return_value = None
    service._process_manager = _LegacyProcessManager(
        running_processes={"Legacy Engine": proc}
    )

    processes = service.get_running_processes()

    assert processes == {
        "Legacy Engine": {"pid": 777, "running": True, "exit_code": None}
    }
    proc.poll.assert_called_once()


@pytest.mark.parametrize(
    ("kill_return", "expected"),
    [
        (True, True),
        (False, False),
        (MagicMock(), True),
    ],
)
def test_stop_process_returns_strict_bool(
    tmp_path: Path, kill_return: object, expected: bool
) -> None:
    """stop_process() must return a strict bool matching kill_process_tree truthiness."""
    service = LauncherService(repo_root=tmp_path)
    proc = MagicMock(pid=1234)
    running_processes = {"proc": proc}
    service._process_manager = MagicMock(running_processes=running_processes)

    mock_module = MagicMock()
    mock_module.kill_process_tree.return_value = kill_return

    with patch.dict(
        sys.modules, {"src.shared.python.security.subprocess_utils": mock_module}
    ):
        result = service.stop_process("proc")

    assert result is expected
    assert type(result) is bool
    mock_module.kill_process_tree.assert_called_once_with(1234)
    assert "proc" not in running_processes
