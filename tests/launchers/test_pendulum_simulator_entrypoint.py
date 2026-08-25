"""Regression coverage for the Pendulum Simulator launcher entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from src.launchers.launcher_model_handlers import SpecialAppHandler

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_pendulum_simulator_module_entrypoint_reports_its_version(
    tmp_path: Path,
) -> None:
    """The launcher-safe module command must start without GUI initialization errors."""
    (tmp_path / "sitecustomize.py").write_text(
        """import importlib.abc
import sys


class _BlockedGuiFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.endswith(".gui.main_window"):
            raise ImportError("version probe imported the GUI")
        return None


sys.meta_path.insert(0, _BlockedGuiFinder())
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(tmp_path),
            str(REPO_ROOT),
            str(REPO_ROOT / "src"),
            environment.get("PYTHONPATH", ""),
        )
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
        keep_terminal_open=True,
    )
    process_manager.launch_script.assert_not_called()


@pytest.mark.unit
def test_pendulum_embed_adapter_resolves_from_tools_provider(tmp_path: Path) -> None:
    """The launcher must load adapters relative to their declared provider."""
    repo_root = tmp_path / "UpstreamDrift"
    repo_root.mkdir()
    tools_root = tmp_path / "Tools"
    adapter = (
        tools_root
        / "src"
        / "pendulum_simulator"
        / "src"
        / "double_pendulum_golf"
        / "_embed_adapter.py"
    )
    adapter.parent.mkdir(parents=True)
    adapter.write_text("def get_dockable_ui():\n    return 'provider-widget'\n")
    model = SimpleNamespace(
        id="pendulum_simulator",
        name="Pendulum Simulator",
        path="src/double_pendulum_golf/__main__.py",
        type="special_app",
        source_root="Tools",
        working_dir="src/pendulum_simulator",
        python_paths=("src/pendulum_simulator/src",),
        embed_adapter=(
            "src/pendulum_simulator/src/double_pendulum_golf/"
            "_embed_adapter.py::get_dockable_ui"
        ),
    )

    result = SpecialAppHandler().get_dockable_ui(model, repo_root)

    assert result == "provider-widget"


@pytest.mark.unit
def test_external_tools_adapter_can_remain_launcher_owned(tmp_path: Path) -> None:
    """Existing Upstream adapters remain valid for Tools-backed models."""
    repo_root = tmp_path / "UpstreamDrift"
    local_adapter = repo_root / "src" / "launchers" / "adapter.py"
    local_adapter.parent.mkdir(parents=True)
    local_adapter.write_text("def get_dockable_ui():\n    return 'local-widget'\n")
    (tmp_path / "Tools").mkdir()
    model = SimpleNamespace(
        id="external_tool",
        source_root="../Tools",
        python_paths=(),
        embed_adapter="src/launchers/adapter.py::get_dockable_ui",
        path="unused.py",
    )

    result = SpecialAppHandler().get_dockable_ui(model, repo_root)

    assert result == "local-widget"


@pytest.mark.unit
def test_pendulum_tile_consumes_canonical_tools_source() -> None:
    """Upstream must not launch its retained compatibility copy by default."""
    config = yaml.safe_load((REPO_ROOT / "src/config/models.yaml").read_text())
    tile = next(item for item in config["models"] if item["id"] == "pendulum_simulator")

    assert tile["provider"] == "tools"
    assert tile["source_root"] == "../Tools/src/pendulum_simulator"
    assert tile["python_paths"] == ["src"]
    assert "src/shared/python/pendulum_simulator" not in tile["path"]
    assert tile["embed_adapter"].endswith("_embed_adapter.py::get_dockable_ui")
