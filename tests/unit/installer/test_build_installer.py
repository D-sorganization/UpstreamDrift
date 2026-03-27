from unittest.mock import MagicMock, patch

import pytest

import installer.windows.build_installer as bi


def test_check_prerequisites(monkeypatch):
    original_import = __import__

    # Test failure
    def mock_import_fail(name, *args, **kwargs):
        if name == "cx_Freeze":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import_fail)
    assert bi.check_prerequisites() is False

    # Test success
    # Need to simulate cx_Freeze existing
    mock_cx = MagicMock()
    mock_cx.version = "1.0.0"

    def mock_import_success(name, *args, **kwargs):
        if name == "cx_Freeze":
            return mock_cx
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import_success)
    assert bi.check_prerequisites() is True


def test_clean_build_dirs(tmp_path, monkeypatch):
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
def test_install_dependencies(mock_run):
    assert bi.install_dependencies() is True
    assert mock_run.call_count == 3


@patch("subprocess.run")
def test_install_dependencies_fail(mock_run):
    from subprocess import CalledProcessError

    mock_run.side_effect = CalledProcessError(1, "cmd")
    assert bi.install_dependencies() is False


def test_detect_physics_engines(monkeypatch):
    def mock_import(name, *args, **kwargs):
        if name in ("mujoco", "pinocchio"):
            return MagicMock()
        raise ImportError

    monkeypatch.setattr("builtins.__import__", mock_import)
    engines = bi.detect_physics_engines()
    assert engines == ["mujoco", "pinocchio"]


@patch("subprocess.run")
@patch("os.chdir")
@patch("os.getcwd", return_value="/tmp")
def test_build_executable(mock_getcwd, mock_chdir, mock_run):
    mock_run.return_value.returncode = 0
    assert bi.build_executable() is True


@patch("subprocess.run")
@patch("os.chdir")
@patch("os.getcwd", return_value="/tmp")
def test_build_msi(mock_getcwd, mock_chdir, mock_run, monkeypatch, tmp_path):
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(bi, "DIST_DIR", tmp_path)
    (tmp_path / "installer.msi").touch()
    assert bi.build_msi() is True


@patch("subprocess.run")
@patch("os.chdir")
@patch("os.getcwd", return_value="/tmp")
def test_build_msi_fail(mock_getcwd, mock_chdir, mock_run):
    mock_run.return_value.returncode = 1
    assert bi.build_msi() is False


def test_create_installer_info(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "DIST_DIR", tmp_path)
    monkeypatch.setattr(bi, "detect_physics_engines", lambda: ["mujoco"])

    bi.create_installer_info()

    info_file = tmp_path / "installer_info.json"
    assert info_file.exists()
    import json

    with open(info_file) as f:
        data = json.load(f)
    assert data["physics_engines"] == ["mujoco"]
    assert "version" in data


@patch("installer.windows.build_installer.check_prerequisites", return_value=True)
@patch("installer.windows.build_installer.clean_build_dirs")
@patch("installer.windows.build_installer.install_dependencies", return_value=True)
@patch(
    "installer.windows.build_installer.detect_physics_engines", return_value=["mujoco"]
)
@patch("installer.windows.build_installer.build_executable", return_value=True)
@patch("installer.windows.build_installer.build_msi", return_value=True)
@patch("installer.windows.build_installer.create_installer_info")
def test_main(
    mock_info,
    mock_msi,
    mock_exe,
    mock_detect,
    mock_deps,
    mock_clean,
    mock_prereq,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(bi, "DIST_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["build_installer.py", "--clean"])

    try:
        bi.main()
    except SystemExit:
        pytest.fail("Main called sys.exit unexpectedly")

    mock_prereq.assert_called_once()
    mock_clean.assert_called_once()
    mock_deps.assert_called_once()
    mock_detect.assert_called_once()
    mock_exe.assert_called_once()
    mock_msi.assert_called_once()
    mock_info.assert_called_once()
