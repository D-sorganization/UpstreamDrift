"""Tests for launcher PYTHONPATH injection and WSL portability (issue #2479).

Launchers that spawn nested subprocesses must inject the repo root into
PYTHONPATH so child scripts can import src.* regardless of the ambient shell
state.  WSL helpers must read distro, project dir, and conda env from
environment variables rather than hard-coded developer-specific paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from src.launchers.launcher_process_manager import ProcessManager


class TestMotionCaptureLauncherPythonpath:
    """motion_capture_launcher._launch_script must inject PYTHONPATH."""

    def test_popen_receives_env_with_pythonpath(self) -> None:
        """_make_subprocess_env must return a dict with repo root in PYTHONPATH."""
        from src.launchers.motion_capture_launcher import (
            REPO_ROOT,
            _make_subprocess_env,
        )

        env = _make_subprocess_env(REPO_ROOT)
        assert "PYTHONPATH" in env, "env dict must contain PYTHONPATH"
        assert (
            str(REPO_ROOT) in env["PYTHONPATH"]
        ), f"REPO_ROOT ({REPO_ROOT}) must appear in PYTHONPATH"

    def test_popen_env_injects_pythonpath_in_subprocess_call(self) -> None:
        """_launch_script must pass env= to subprocess.Popen."""
        import inspect

        import src.launchers.motion_capture_launcher as mod

        src_text = inspect.getsource(mod)
        # The Popen call must pass env= keyword argument
        assert (
            "env=" in src_text
        ), "subprocess.Popen(...) call must pass env= to inject PYTHONPATH"


class TestGolfSuiteLauncherPythonpath:
    """golf_suite_launcher._launch_script must inject PYTHONPATH."""

    def test_popen_env_injects_pythonpath_in_subprocess_call(self) -> None:
        """_launch_script must pass env= to subprocess.Popen."""

        # We read source rather than importing, as PyQt6 may not be installed.
        launcher_src = (
            Path(__file__).parents[2] / "src" / "launchers" / "golf_suite_launcher.py"
        )
        src_text = launcher_src.read_text(encoding="utf-8")
        # The Popen call in _launch_script must pass env=
        assert (
            "env=" in src_text
        ), "subprocess.Popen(...) in _launch_script must pass env= to inject PYTHONPATH"


class TestProcessManagerWslConfig:
    """launch_in_wsl / launch_module_in_wsl must read settings from env vars."""

    def _make_manager(self) -> ProcessManager:
        from src.launchers.launcher_process_manager import ProcessManager

        return ProcessManager(repo_root=Path("/tmp/fake_repo"))

    def test_wsl_distro_reads_from_env(self) -> None:
        """WSL distro must be read from WSL_DISTRO env var, not hard-coded."""
        with patch.dict(os.environ, {"WSL_DISTRO": "MyDistro"}, clear=False):
            mgr = self._make_manager()
            distro = mgr._get_wsl_distro()
            assert (
                distro == "MyDistro"
            ), f"Expected distro 'MyDistro' from env, got '{distro}'"

    def test_wsl_distro_defaults_to_fallback_when_unset(self) -> None:
        """WSL distro must have a safe fallback when WSL_DISTRO is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WSL_DISTRO", None)
            mgr = self._make_manager()
            distro = mgr._get_wsl_distro()
            assert (
                isinstance(distro, str) and len(distro) > 0
            ), "_get_wsl_distro must return a non-empty string fallback"

    def test_wsl_project_dir_reads_from_env(self) -> None:
        """WSL project dir must be read from WSL_PROJECT_DIR env var."""
        with patch.dict(os.environ, {"WSL_PROJECT_DIR": "/mnt/d/MyRepo"}, clear=False):
            mgr = self._make_manager()
            proj_dir = mgr._get_wsl_project_dir()
            assert proj_dir == "/mnt/d/MyRepo"

    def test_wsl_conda_env_reads_from_env(self) -> None:
        """Conda env name must be read from WSL_CONDA_ENV env var."""
        with patch.dict(os.environ, {"WSL_CONDA_ENV": "my_env"}, clear=False):
            mgr = self._make_manager()
            conda_env = mgr._get_wsl_conda_env()
            assert conda_env == "my_env"

    def test_wsl_conda_env_defaults_to_fallback_when_unset(self) -> None:
        """Conda env must have a safe fallback when WSL_CONDA_ENV is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WSL_CONDA_ENV", None)
            mgr = self._make_manager()
            conda_env = mgr._get_wsl_conda_env()
            assert isinstance(conda_env, str) and len(conda_env) > 0

    def test_no_hardcoded_developer_path_in_source(self) -> None:
        """Hard-coded developer path /mnt/c/Users/diete must not appear in source."""
        src_file = (
            Path(__file__).parents[2]
            / "src"
            / "launchers"
            / "launcher_process_manager.py"
        )
        src_text = src_file.read_text(encoding="utf-8")
        assert "/mnt/c/Users/diete" not in src_text, (
            "Hard-coded developer path '/mnt/c/Users/diete' found in source. "
            "Use WSL_PROJECT_DIR env var instead."
        )

    def test_no_hardcoded_distro_in_source(self) -> None:
        """Hard-coded distro 'Ubuntu-22.04' must not appear in source."""
        src_file = (
            Path(__file__).parents[2]
            / "src"
            / "launchers"
            / "launcher_process_manager.py"
        )
        src_text = src_file.read_text(encoding="utf-8")
        assert "Ubuntu-22.04" not in src_text, (
            "Hard-coded distro 'Ubuntu-22.04' found in source. "
            "Use WSL_DISTRO env var instead."
        )

    def test_no_hardcoded_conda_env_in_source(self) -> None:
        """Hard-coded conda env 'golf_suite' in WSL command must not appear in source."""
        src_file = (
            Path(__file__).parents[2]
            / "src"
            / "launchers"
            / "launcher_process_manager.py"
        )
        src_text = src_file.read_text(encoding="utf-8")
        # The string 'golf_suite' was the hard-coded conda env name
        assert "conda activate golf_suite" not in src_text, (
            "Hard-coded conda env 'golf_suite' found in source. "
            "Use WSL_CONDA_ENV env var instead."
        )
