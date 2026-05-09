from __future__ import annotations

import pathlib
from unittest.mock import patch

import setup_golf_suite
from PIL import Image


def test_apply_icon_optimizations() -> None:
    # Create dummy image
    img = Image.new("RGBA", (32, 32))
    opt = setup_golf_suite._apply_icon_optimizations(img, 32)
    assert opt is not None


def test_create_optimized_icon(tmp_path) -> None:
    img = Image.new("RGBA", (256, 256))
    src = tmp_path / "src.png"
    img.save(src)

    out = tmp_path / "out.ico"
    res = setup_golf_suite.create_optimized_icon(src, out)
    assert res is True
    assert out.exists()


def test_create_optimized_icon_missing() -> None:
    res = setup_golf_suite.create_optimized_icon(
        pathlib.Path("missing.png"), pathlib.Path("out.ico")
    )
    assert res is False


@patch("subprocess.run")
def test_create_shortcut_windows(mock_run) -> None:
    res = setup_golf_suite.create_shortcut_windows(
        "script.py", pathlib.Path("/wd"), pathlib.Path("/icon.ico"), "desc"
    )
    assert res is True
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_create_shortcut_windows_fail(mock_run) -> None:
    from subprocess import CalledProcessError

    mock_run.side_effect = CalledProcessError(1, "cmd", b"", b"error")
    res = setup_golf_suite.create_shortcut_windows(
        "script.py", pathlib.Path("/wd"), pathlib.Path("/icon.ico"), "desc"
    )
    assert res is False


def test_find_source_image(tmp_path) -> None:
    res = setup_golf_suite._find_source_image(tmp_path)
    assert res is None

    src = tmp_path / "src" / "launchers" / "assets" / "golf_robot_cropped.png"
    src.parent.mkdir(parents=True)
    src.touch()
    res = setup_golf_suite._find_source_image(tmp_path)
    assert res == src


@patch("setup_golf_suite.git_sync_repository")
@patch("setup_golf_suite.check_python_dependencies")
@patch("setup_golf_suite.create_optimized_icon")
def test_setup_golf_suite_main(
    mock_create_icon, mock_chk_deps, mock_sync, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("setup_golf_suite.get_repo_root", lambda: tmp_path)
    mock_chk_deps.return_value = True

    source = tmp_path / "src" / "launchers" / "assets" / "golf_robot_icon.png"
    source.parent.mkdir(parents=True)
    source.touch()

    # Needs to not attempt creation of actual shortcuts on CI, but we mock the platform or function.
    monkeypatch.setattr(
        "setup_golf_suite.create_shortcut_windows", lambda *args, **kwargs: True
    )

    assert setup_golf_suite.main() == 0
    mock_create_icon.assert_called_once_with(
        source, tmp_path / "src" / "launchers" / "assets" / "golf_suite_unified.ico"
    )


@patch("setup_golf_suite.git_sync_repository")
@patch("setup_golf_suite.check_python_dependencies")
def test_main_no_deps(mock_chk_deps, mock_sync) -> None:
    mock_chk_deps.return_value = False
    assert setup_golf_suite.main() == 1
