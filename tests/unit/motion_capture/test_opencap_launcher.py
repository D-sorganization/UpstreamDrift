"""Unit tests for the OpenCap sidecar runner (#11406).

Tests launcher discovery, ADR-0053 licensing and privacy constraints,
fail-closed error handling with install hints, command building, and
output collection via load_opencap_session.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from motion_capture.opencap_ingest.launcher import (
    OpenCapLaunchConfig,
    OpenCapLauncher,
    OpenCapLaunchResult,
    OpenCapSidecarNotFoundError,
    run_opencap_sidecar,
)
from motion_capture.opencap_ingest.output_adapter import OpenCapOutputAdapter
from tests.unit.motion_pipeline.sources.opencap_fixtures import (
    write_opencap_session,
    write_scaled_model,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Dataclass Defaults & Basic Config
# ---------------------------------------------------------------------------


def test_launch_config_defaults(tmp_path: Path) -> None:
    cfg = OpenCapLaunchConfig(session_dir=tmp_path)
    assert cfg.session_dir == tmp_path
    assert cfg.video_dir is None
    assert cfg.calibration_dir is None
    assert cfg.opencap_env is None
    assert cfg.docker_image == "opencap/core:latest"
    assert cfg.runner_type == "auto"
    assert cfg.detector == "hrnet"  # ADR-0053 commercial default
    assert cfg.allow_non_commercial is False
    assert cfg.timeout_seconds == 3600
    assert cfg.trial_name is None
    assert cfg.extra_args is None
    assert cfg.dry_run is False


def test_launch_result_defaults(tmp_path: Path) -> None:
    res = OpenCapLaunchResult(
        success=True,
        return_code=0,
        session_dir=tmp_path,
    )
    assert res.success is True
    assert res.return_code == 0
    assert res.session_dir == tmp_path
    assert res.session is None
    assert res.log_file is None
    assert res.error_message is None
    assert res.used_docker is False
    assert res.used_real_opencap is False


# ---------------------------------------------------------------------------
# ADR-0053 Licensing Invariant
# ---------------------------------------------------------------------------


def test_openpose_detector_requires_explicit_opt_in(tmp_path: Path) -> None:
    """ADR-0053: OpenPose has a non-commercial license and must be opt-in."""
    launcher = OpenCapLauncher()
    cfg = OpenCapLaunchConfig(
        session_dir=tmp_path,
        detector="openpose",
        allow_non_commercial=False,
    )
    with pytest.raises(ValueError, match="non-commercial"):
        launcher.launch(cfg)


def test_openpose_detector_allowed_with_explicit_opt_in(tmp_path: Path) -> None:
    """ADR-0053: OpenPose allowed when allow_non_commercial is True."""
    launcher = OpenCapLauncher()
    cfg = OpenCapLaunchConfig(
        session_dir=tmp_path,
        detector="openpose",
        allow_non_commercial=True,
        dry_run=True,
    )
    res = launcher.launch(cfg)
    assert res.success is True


# ---------------------------------------------------------------------------
# Fail Closed With Install Hint
# ---------------------------------------------------------------------------


def test_fails_closed_with_install_hint_when_sidecar_absent(tmp_path: Path) -> None:
    """Acceptance: Fails closed with an install hint when the sidecar is absent."""
    launcher = OpenCapLauncher()
    cfg = OpenCapLaunchConfig(
        session_dir=tmp_path,
        runner_type="subprocess",
    )
    with (
        patch.object(launcher, "find_opencap_python", return_value=None),
        patch.object(launcher, "is_docker_available", return_value=False),
    ):
        res = launcher.launch(cfg)
        assert res.success is False
        assert res.return_code != 0
        assert res.error_message is not None
        assert "opencap-core" in res.error_message
        assert "github.com/stanfordnmbl/opencap-core" in res.error_message


def test_raises_sidecar_not_found_when_requested(tmp_path: Path) -> None:
    launcher = OpenCapLauncher()
    cfg = OpenCapLaunchConfig(
        session_dir=tmp_path,
        runner_type="subprocess",
    )
    with (
        patch.object(launcher, "find_opencap_python", return_value=None),
        patch.object(launcher, "is_docker_available", return_value=False),
    ):
        with pytest.raises(OpenCapSidecarNotFoundError, match="install"):
            launcher.launch(cfg, raise_on_error=True)


# ---------------------------------------------------------------------------
# Environment & Python Discovery
# ---------------------------------------------------------------------------


def test_find_python_with_explicit_env_path(tmp_path: Path) -> None:
    env = tmp_path / "custom-opencap"
    bin_dir = env / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python"
    py.write_text("#!/bin/sh\n")

    launcher = OpenCapLauncher()
    found = launcher.find_opencap_python(env)
    assert found == str(py)


def test_find_python_with_windows_scripts(tmp_path: Path) -> None:
    env = tmp_path / "custom-opencap-win"
    scripts_dir = env / "Scripts"
    scripts_dir.mkdir(parents=True)
    py = scripts_dir / "python.exe"
    py.write_text("")

    launcher = OpenCapLauncher()
    found = launcher.find_opencap_python(env)
    assert found == str(py)


def test_find_python_via_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    py = tmp_path / "env_var_python"
    py.write_text("")
    monkeypatch.setenv("OPENCAP_PYTHON", str(py))

    launcher = OpenCapLauncher()
    assert launcher.find_opencap_python(None) == str(py)


def test_find_python_via_conda_default(tmp_path: Path) -> None:
    conda_env = tmp_path / "miniconda3" / "envs" / "opencap-core"
    (conda_env / "bin").mkdir(parents=True)
    py = conda_env / "bin" / "python"
    py.write_text("")

    launcher = OpenCapLauncher()
    with patch(
        "motion_capture.opencap_ingest.launcher.Path.home", return_value=tmp_path
    ):
        assert launcher.find_opencap_python(None) == str(py)


# ---------------------------------------------------------------------------
# Command Building
# ---------------------------------------------------------------------------


def test_build_subprocess_command(tmp_path: Path) -> None:
    launcher = OpenCapLauncher()
    cfg = OpenCapLaunchConfig(
        session_dir=tmp_path / "session",
        video_dir=tmp_path / "videos",
        calibration_dir=tmp_path / "calib",
        detector="hrnet",
        extra_args=["--filter", "butterworth"],
    )
    cmd = launcher.build_command(cfg, python_exe="/opt/opencap/bin/python")
    assert cmd[0] == "/opt/opencap/bin/python"
    assert "-m" in cmd
    assert "opencap_core" in cmd
    assert "--session_dir" in cmd
    assert str(tmp_path / "session") in cmd
    assert "--video_dir" in cmd
    assert str(tmp_path / "videos") in cmd
    assert "--calibration_dir" in cmd
    assert str(tmp_path / "calib") in cmd
    assert "--detector" in cmd
    assert "hrnet" in cmd
    assert "--filter" in cmd
    assert "butterworth" in cmd


def test_build_docker_command(tmp_path: Path) -> None:
    launcher = OpenCapLauncher()
    cfg = OpenCapLaunchConfig(
        session_dir=tmp_path / "session",
        video_dir=tmp_path / "videos",
        calibration_dir=tmp_path / "calib",
        docker_image="custom/opencap:v1",
        detector="mmpose",
    )
    cmd = launcher.build_docker_command(cfg)
    assert cmd[0] == "docker"
    assert "run" in cmd
    assert "--rm" in cmd
    assert "-v" in cmd
    assert "custom/opencap:v1" in cmd


# ---------------------------------------------------------------------------
# Dry-Run Execution & Session Collection
# ---------------------------------------------------------------------------


def test_dry_run_creates_valid_session_and_collects(tmp_path: Path) -> None:
    launcher = OpenCapLauncher()
    session_dir = tmp_path / "dry_run_session"
    cfg = OpenCapLaunchConfig(
        session_dir=session_dir,
        dry_run=True,
    )
    res = launcher.launch(cfg)
    assert res.success is True
    assert res.return_code == 0
    assert res.session is not None
    assert res.session.trial == "trial1"
    assert res.session.model_file is not None
    assert res.session.model_file.exists()


# ---------------------------------------------------------------------------
# Subprocess Execution via managed_popen & Output Collection
# ---------------------------------------------------------------------------


def test_successful_subprocess_execution_collects_session(tmp_path: Path) -> None:
    launcher = OpenCapLauncher()
    session_dir = tmp_path / "mock_session"
    session_dir.mkdir(parents=True)

    cfg = OpenCapLaunchConfig(
        session_dir=session_dir,
        runner_type="subprocess",
    )

    # Mock process object
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.returncode = 0

    def fake_managed_popen(*args, **kwargs):
        class Ctx:
            def __enter__(self):
                # Write session files as if opencap-core ran
                s = write_opencap_session(session_dir.parent, trials=("trial1",))
                write_scaled_model(s)
                # move to session_dir
                for item in s.iterdir():
                    dest = session_dir / item.name
                    if item.is_dir():
                        item.rename(dest)
                    else:
                        item.rename(dest)
                return mock_proc

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

        return Ctx()

    with (
        patch.object(launcher, "find_opencap_python", return_value="/mock/python"),
        patch(
            "motion_capture.opencap_ingest.launcher.managed_popen",
            side_effect=fake_managed_popen,
        ),
    ):
        res = launcher.launch(cfg)
        assert res.success is True
        assert res.return_code == 0
        assert res.session is not None
        assert res.session.trial == "trial1"
        assert res.session.observations is not None


def test_subprocess_non_zero_exit_returns_error(tmp_path: Path) -> None:
    launcher = OpenCapLauncher()
    session_dir = tmp_path / "err_session"
    session_dir.mkdir(parents=True)

    cfg = OpenCapLaunchConfig(
        session_dir=session_dir,
        runner_type="subprocess",
    )

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.returncode = 1

    def fake_managed_popen(*args, **kwargs):
        class Ctx:
            def __enter__(self):
                # Write some log content
                log_file = kwargs.get("stdout")
                if log_file:
                    log_file.write("RuntimeError: GPU out of memory\n")
                    log_file.flush()
                return mock_proc

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

        return Ctx()

    with (
        patch.object(launcher, "find_opencap_python", return_value="/mock/python"),
        patch(
            "motion_capture.opencap_ingest.launcher.managed_popen",
            side_effect=fake_managed_popen,
        ),
    ):
        res = launcher.launch(cfg)
        assert res.success is False
        assert res.return_code == 1
        assert "exited with code 1" in res.error_message


# ---------------------------------------------------------------------------
# Output Adapter
# ---------------------------------------------------------------------------


def test_opencap_output_adapter_load_and_inspect(tmp_path: Path) -> None:
    session_dir = write_opencap_session(tmp_path, trials=("trial1",))
    write_scaled_model(session_dir)

    adapter = OpenCapOutputAdapter()
    session = adapter.load(session_dir)
    assert session.trial == "trial1"
    assert session.model_file is not None

    meta = adapter.inspect(session_dir)
    assert meta.trials == ["trial1"]
    assert meta.model_file is not None
