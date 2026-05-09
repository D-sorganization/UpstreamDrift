from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import installer.windows.build_installer as bi
import pytest


def test_check_prerequisites(monkeypatch) -> None:
    original_import = __import__

    # Test failure
    def mock_import_fail(name, *args, **kwargs) -> Any:
        if name == "cx_Freeze":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import_fail)
    assert bi.check_prerequisites() is False

    # Test success
    # Need to simulate cx_Freeze existing
    mock_cx = MagicMock()
    mock_cx.version = "1.0.0"

    def mock_import_success(name, *args, **kwargs) -> Any:
        if name == "cx_Freeze":
            return mock_cx
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import_success)
    assert bi.check_prerequisites() is True


def test_check_prerequisites_logs_cx_freeze_version(monkeypatch, caplog) -> None:
    original_import = __import__
    mock_cx = MagicMock()
    mock_cx.version = "9.9.9"

    def mock_import_success(name, *args, **kwargs) -> Any:
        if name == "cx_Freeze":
            return mock_cx
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import_success)

    with caplog.at_level("INFO"):
        assert bi.check_prerequisites() is True

    assert "cx_Freeze 9.9.9" in caplog.text


def test_clean_build_dirs(tmp_path, monkeypatch) -> None:
    build_dir = tmp_path / "build"
    dist_dir = tmp_path / "dist"

    monkeypatch.setattr(bi, "BUILD_DIR", build_dir)
    monkeypatch.setattr(bi, "DIST_DIR", dist_dir)

    build_dir.mkdir()
    (build_dir / "old_file.txt").touch()

    bi.clean_build_dirs()

    assert build_dir.exists()
    assert dist_dir.exists()
    assert not (build_dir / "old_file.txt").exists()


@patch("subprocess.run")
def test_install_dependencies(mock_run) -> None:
    assert bi.install_dependencies() is True
    assert mock_run.call_count == 3


@patch("subprocess.run")
def test_install_dependencies_fail(mock_run) -> None:
    from subprocess import CalledProcessError

    mock_run.side_effect = CalledProcessError(1, "cmd")
    assert bi.install_dependencies() is False


def test_detect_physics_engines(monkeypatch) -> None:
    def mock_import(name, *args, **kwargs) -> Any:
        if name in ("mujoco", "pinocchio"):
            return MagicMock()
        raise ImportError

    monkeypatch.setattr("builtins.__import__", mock_import)
    engines = bi.detect_physics_engines()
    assert engines == ["mujoco", "pinocchio"]


@patch("subprocess.run")
@patch("os.chdir")
@patch("os.getcwd", return_value="/tmp")
def test_build_executable(mock_getcwd, mock_chdir, mock_run) -> None:
    mock_run.return_value.returncode = 0
    assert bi.build_executable("hybrid") is True
    assert (
        mock_run.call_args.kwargs["env"]["UPSTREAM_DRIFT_INSTALL_PROFILE"] == "hybrid"
    )


@patch("subprocess.run")
@patch("os.chdir")
@patch("os.getcwd", return_value="/tmp")
def test_build_msi(mock_getcwd, mock_chdir, mock_run, monkeypatch, tmp_path) -> None:
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(bi, "DIST_DIR", tmp_path)
    (tmp_path / "installer.msi").touch()
    assert bi.build_msi("full", ("C:/repos/Drake_Models",)) is True
    assert mock_run.call_args.kwargs["env"]["UPSTREAM_DRIFT_INSTALL_PROFILE"] == "full"
    assert mock_run.call_args.kwargs["env"]["UPSTREAM_DRIFT_PROVIDER_ROOTS"] == str(
        Path("C:/repos/Drake_Models")
    )


@patch("subprocess.run")
@patch("os.chdir")
@patch("os.getcwd", return_value="/tmp")
def test_build_msi_fail(mock_getcwd, mock_chdir, mock_run) -> None:
    mock_run.return_value.returncode = 1
    assert bi.build_msi("hybrid") is False


def test_create_installer_info(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bi, "DIST_DIR", tmp_path)
    monkeypatch.setattr(bi, "detect_physics_engines", lambda: ["mujoco"])

    bi.create_installer_info("full", ("C:/repos/MuJoCo_Models",))

    info_file = tmp_path / "installer_info.json"
    assert info_file.exists()
    import json

    with open(info_file) as f:
        data = json.load(f)
    assert data["physics_engines"] == ["mujoco"]
    assert "version" in data
    assert data["packaging_profile"] == "full"
    assert data["discovery_mode"] == "provider-first"
    assert data["provider_roots"] == [str(Path("C:/repos/MuJoCo_Models"))]


def test_log_generated_outputs(caplog, tmp_path) -> None:
    artifact = tmp_path / "installer.msi"
    artifact.write_text("artifact")

    with caplog.at_level("INFO"):
        bi._log_generated_outputs([artifact])

    assert "Generated installer.msi" in caplog.text


@patch("installer.windows.build_installer.check_prerequisites", return_value=True)
@patch("installer.windows.build_installer.clean_build_dirs")
@patch("installer.windows.build_installer.install_dependencies", return_value=True)
@patch(
    "installer.windows.build_installer.detect_physics_engines", return_value=["mujoco"]
)
@patch("installer.windows.build_installer.build_executable", return_value=True)
@patch("installer.windows.build_installer.build_msi", return_value=True)
@patch("installer.windows.build_installer.create_installer_info")
@patch("installer.windows.build_installer._log_generated_outputs")
def test_build_installer_main(
    mock_log_outputs,
    mock_info,
    mock_msi,
    mock_exe,
    mock_detect,
    mock_deps,
    mock_clean,
    mock_prereq,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(bi, "DIST_DIR", tmp_path)
    artifact = tmp_path / "artifact.msi"
    artifact.write_bytes(b"abc")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_installer.py",
            "--clean",
            "--profile",
            "full",
            "--provider-root",
            "C:/repos/MuJoCo_Models",
        ],
    )
    (tmp_path / "installer.msi").write_text("artifact")

    try:
        bi.main()
    except SystemExit:
        pytest.fail("Main called sys.exit unexpectedly")

    mock_prereq.assert_called_once()
    mock_clean.assert_called_once()
    mock_deps.assert_called_once()
    mock_detect.assert_called_once()
    mock_exe.assert_called_once_with("full", ("C:/repos/MuJoCo_Models",))
    mock_msi.assert_called_once_with("full", ("C:/repos/MuJoCo_Models",))
    mock_info.assert_called_once_with("full", ("C:/repos/MuJoCo_Models",))
    mock_log_outputs.assert_called_once()
