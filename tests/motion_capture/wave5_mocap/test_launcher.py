"""Fast tests for motion_capture.freemocap_ingest.launcher."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from motion_capture.freemocap_ingest.launcher import (
    FreeMoCapLauncher,
    LaunchConfig,
    LaunchResult,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_launch_config_defaults(tmp_path: Path) -> None:
    cfg = LaunchConfig(session_dir=tmp_path)
    assert cfg.freemocap_env is None
    assert cfg.timeout_seconds == 3600
    assert cfg.video_dir is None
    assert cfg.output_dir is None
    assert cfg.headless is True
    assert cfg.extra_args is None


def test_launcher_default_log_level() -> None:
    launcher = FreeMoCapLauncher()
    assert launcher.log_level == logging.INFO


# ---------------------------------------------------------------------------
# _find_freemocap_python
# ---------------------------------------------------------------------------


def test_find_python_with_env_path_existing(tmp_path: Path) -> None:
    env = tmp_path / "freemocap-env"
    (env / "bin").mkdir(parents=True)
    py = env / "bin" / "python"
    py.write_text("#!/bin/sh\n")
    launcher = FreeMoCapLauncher()
    found = launcher._find_freemocap_python(env)
    assert found == str(py)


def test_find_python_env_path_missing_falls_through(tmp_path: Path) -> None:
    # env path provided but doesn't contain a python; should fall to search
    env = tmp_path / "missing-env"
    launcher = FreeMoCapLauncher()
    with patch(
        "motion_capture.freemocap_ingest.launcher.Path.home", return_value=tmp_path
    ):
        result = launcher._find_freemocap_python(env)
    assert result is None


def test_find_python_via_conda_default(tmp_path: Path) -> None:
    conda_env = tmp_path / "miniconda3" / "envs" / "freemocap-env"
    (conda_env / "bin").mkdir(parents=True)
    py = conda_env / "bin" / "python"
    py.write_text("")
    launcher = FreeMoCapLauncher()
    with patch(
        "motion_capture.freemocap_ingest.launcher.Path.home", return_value=tmp_path
    ):
        assert launcher._find_freemocap_python(None) == str(py)


def test_find_python_via_dot_venvs(tmp_path: Path) -> None:
    venv = tmp_path / ".venvs" / "freemocap-env"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python"
    py.write_text("")
    launcher = FreeMoCapLauncher()
    with patch(
        "motion_capture.freemocap_ingest.launcher.Path.home", return_value=tmp_path
    ):
        assert launcher._find_freemocap_python(None) == str(py)


def test_find_python_none_when_not_found(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    with (
        patch(
            "motion_capture.freemocap_ingest.launcher.Path.home", return_value=tmp_path
        ),
        patch.object(Path, "exists", return_value=False),
    ):
        assert launcher._find_freemocap_python(None) is None


# ---------------------------------------------------------------------------
# _build_command
# ---------------------------------------------------------------------------


def test_build_command_minimal(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    cfg = LaunchConfig(session_dir=tmp_path / "session_x")
    cmd = launcher._build_command(cfg, "/usr/bin/python")
    assert cmd[:3] == ["/usr/bin/python", "-m", "freemocap"]
    assert "--headless" in cmd
    assert "--session_id" in cmd
    assert "session_x" in cmd
    assert "--output_path" in cmd


def test_build_command_with_all_options(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    cfg = LaunchConfig(
        session_dir=tmp_path / "s",
        video_dir=tmp_path / "videos",
        output_dir=tmp_path / "out",
        headless=False,
        extra_args=["--gpu", "1"],
    )
    cmd = launcher._build_command(cfg, "python3")
    assert "--headless" not in cmd
    assert "--video_path" in cmd
    assert str(tmp_path / "videos") in cmd
    assert str(tmp_path / "out") in cmd
    assert "--gpu" in cmd and "1" in cmd


# ---------------------------------------------------------------------------
# _setup_logging
# ---------------------------------------------------------------------------


def test_setup_logging_creates_log_dir(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    log_file = launcher._setup_logging(tmp_path)
    assert log_file.parent.name == "logs"
    assert log_file.parent.exists()
    assert log_file.name.startswith("freemocap_")
    assert log_file.suffix == ".log"


# ---------------------------------------------------------------------------
# launch()
# ---------------------------------------------------------------------------


def test_launch_session_dir_missing(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    cfg = LaunchConfig(session_dir=tmp_path / "absent")
    result = launcher.launch(cfg)
    assert isinstance(result, LaunchResult)
    assert result.success is False
    assert result.return_code == -1
    assert "does not exist" in result.error_message


def test_launch_env_not_found(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    with patch.object(launcher, "_find_freemocap_python", return_value=None):
        result = launcher.launch(LaunchConfig(session_dir=tmp_path))
    assert result.success is False
    assert "FreeMoCap environment not found" in result.error_message


def _make_popen(return_code: int = 0, timeout: bool = False) -> MagicMock:
    proc = MagicMock()
    proc.pid = 4242
    if timeout:
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=1), 0]
    else:
        proc.wait.return_value = return_code
    return proc


def test_launch_success(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    with (
        patch.object(launcher, "_find_freemocap_python", return_value="/fake/python"),
        patch("subprocess.Popen", return_value=_make_popen(0)) as popen,
    ):
        result = launcher.launch(LaunchConfig(session_dir=tmp_path))

    assert result.success is True
    assert result.return_code == 0
    assert result.output_dir == tmp_path.resolve() / launcher.DEFAULT_OUTPUT_SUBDIR
    assert result.log_file is not None and result.log_file.exists()
    popen.assert_called_once()


def test_launch_nonzero_exit(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    with (
        patch.object(launcher, "_find_freemocap_python", return_value="/fake/python"),
        patch("subprocess.Popen", return_value=_make_popen(3)),
    ):
        result = launcher.launch(LaunchConfig(session_dir=tmp_path))

    assert result.success is False
    assert result.return_code == 3
    assert "exited with code 3" in result.error_message


def test_launch_timeout(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    proc = _make_popen(timeout=True)
    with (
        patch.object(launcher, "_find_freemocap_python", return_value="/fake/python"),
        patch("subprocess.Popen", return_value=proc),
    ):
        result = launcher.launch(LaunchConfig(session_dir=tmp_path, timeout_seconds=1))
    assert result.success is False
    assert "timed out" in result.error_message
    proc.kill.assert_called_once()


def test_launch_subprocess_exception(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    with (
        patch.object(launcher, "_find_freemocap_python", return_value="/fake/python"),
        patch("subprocess.Popen", side_effect=OSError("boom")),
    ):
        result = launcher.launch(LaunchConfig(session_dir=tmp_path))
    assert result.success is False
    assert "boom" in result.error_message


def test_launch_respects_explicit_output_dir(tmp_path: Path) -> None:
    launcher = FreeMoCapLauncher()
    out = tmp_path / "custom_out"
    with (
        patch.object(launcher, "_find_freemocap_python", return_value="/fake/python"),
        patch("subprocess.Popen", return_value=_make_popen(0)),
    ):
        result = launcher.launch(LaunchConfig(session_dir=tmp_path, output_dir=out))
    assert result.success is True
    assert result.output_dir == out
