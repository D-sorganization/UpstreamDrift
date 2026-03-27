from __future__ import annotations

import pathlib
from unittest.mock import patch

from PIL import Image

import setup_golf_suite


def test_apply_icon_optimizations():
    # Create dummy image
    img = Image.new("RGBA", (32, 32))
    opt = setup_golf_suite._apply_icon_optimizations(img, 32)
    assert opt is not None


def test_create_optimized_icon(tmp_path):
    img = Image.new("RGBA", (256, 256))
    src = tmp_path / "src.png"
    img.save(src)

    out = tmp_path / "out.ico"
    res = setup_golf_suite.create_optimized_icon(src, out)
    assert res is True
    assert out.exists()


def test_create_optimized_icon_missing():
    res = setup_golf_suite.create_optimized_icon(
        pathlib.Path("missing.png"), pathlib.Path("out.ico")
    )
    assert res is False


@patch("subprocess.run")
def test_create_shortcut_windows(mock_run):
    res = setup_golf_suite.create_shortcut_windows(
        "script.py", pathlib.Path("/wd"), pathlib.Path("/icon.ico"), "desc"
    )
    assert res is True
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_create_shortcut_windows_fail(mock_run):
    from subprocess import CalledProcessError

    mock_run.side_effect = CalledProcessError(1, "cmd", b"", b"error")
    res = setup_golf_suite.create_shortcut_windows(
        "script.py", pathlib.Path("/wd"), pathlib.Path("/icon.ico"), "desc"
    )
    assert res is False


def test_find_source_image(tmp_path):
    res = setup_golf_suite._find_source_image(tmp_path)
    assert res is None

    src = tmp_path / "GolfingRobot.png"
    src.touch()
    res = setup_golf_suite._find_source_image(tmp_path)
    assert res == src


@patch("setup_golf_suite.git_sync_repository")
@patch("setup_golf_suite.check_python_dependencies")
@patch("setup_golf_suite.create_optimized_icon")
def test_main(mock_create_icon, mock_chk_deps, mock_sync, tmp_path, monkeypatch):
    monkeypatch.setattr("setup_golf_suite.get_repo_root", lambda: tmp_path)
    mock_chk_deps.return_value = True

    # Needs to not attempt creation of actual shortcuts on CI, but we mock the platform or function.
    monkeypatch.setattr(
        "setup_golf_suite.create_shortcut_windows", lambda *args, **kwargs: True
    )

    assert setup_golf_suite.main() == 0


@patch("setup_golf_suite.git_sync_repository")
@patch("setup_golf_suite.check_python_dependencies")
def test_main_no_deps(mock_chk_deps, mock_sync):
    mock_chk_deps.return_value = False
    assert setup_golf_suite.main() == 1
