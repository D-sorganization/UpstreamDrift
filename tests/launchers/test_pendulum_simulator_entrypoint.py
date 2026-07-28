"""Regression coverage for the Pendulum Simulator launcher entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from src.launchers.launcher_model_handlers import SpecialAppHandler


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_pendulum_simulator_module_entrypoint_reports_its_version() -> None:
    """The launcher-safe module command must start without GUI initialization errors."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), str(REPO_ROOT / "src"), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shared.python.pendulum_simulator",
            "--version",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith("pendulum-simulator ")


@pytest.mark.unit
def test_pendulum_tile_uses_module_launch_not_script_launch(tmp_path: Path) -> None:
    """Package-relative Pendulum imports require the ``python -m`` launch form."""
    entrypoint = (
        tmp_path / "src" / "shared" / "python" / "pendulum_simulator" / "__main__.py"
    )
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("pass\n", encoding="utf-8")
    model = SimpleNamespace(
        id="pendulum_simulator",
        name="Pendulum Simulator",
        path="src/shared/python/pendulum_simulator/__main__.py",
        type="special_app",
        source_root=None,
        working_dir=None,
        python_paths=(),
    )
    process_manager = MagicMock()
    process_manager.launch_module.return_value = MagicMock()

    assert SpecialAppHandler().launch(model, tmp_path, process_manager) is True

    process_manager.launch_module.assert_called_once_with(
        name="Pendulum Simulator",
        module_name="shared.python.pendulum_simulator",
        cwd=tmp_path,
        extra_python_paths=(),
        confirm_startup=True,
    )
    process_manager.launch_script.assert_not_called()


@pytest.mark.unit
def test_pendulum_tile_reports_fast_module_startup_failure(tmp_path: Path) -> None:
    """A child process that dies during startup must not produce a success toast."""
    entrypoint = (
        tmp_path / "src" / "shared" / "python" / "pendulum_simulator" / "__main__.py"
    )
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("raise SystemExit(1)\n", encoding="utf-8")
    model = SimpleNamespace(
        id="pendulum_simulator",
        name="Pendulum Simulator",
        path="src/shared/python/pendulum_simulator/__main__.py",
        type="special_app",
        source_root=None,
        working_dir=None,
        python_paths=(),
    )
    process_manager = MagicMock()
    process_manager.launch_module.return_value = None

    assert SpecialAppHandler().launch(model, tmp_path, process_manager) is False

    process_manager.launch_module.assert_called_once_with(
        name="Pendulum Simulator",
        module_name="shared.python.pendulum_simulator",
        cwd=tmp_path,
        extra_python_paths=(),
        confirm_startup=True,
    )
    process_manager.launch_script.assert_not_called()
