"""Comprehensive tests for ``installer.legacy_setup`` (wave 9)."""

from __future__ import annotations

import pathlib
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from installer import legacy_setup


# ---------------------------------------------------------------------------
# _apply_icon_optimizations
# ---------------------------------------------------------------------------


class _FakeImage:
    """Stand-in PIL image that records filters/enhancement calls."""

    def __init__(self) -> None:
        self.filters: list[object] = []

    def filter(self, flt: object) -> _FakeImage:  # noqa: A003 - PIL API
        self.filters.append(flt)
        return self


class TestApplyIconOptimizations:
    def test_rejects_none_image(self) -> None:
        with pytest.raises(ValueError, match="Image cannot be None"):
            legacy_setup._apply_icon_optimizations(None, 32)  # type: ignore[arg-type]

    def test_rejects_none_size(self) -> None:
        with pytest.raises(ValueError, match="size must be provided"):
            legacy_setup._apply_icon_optimizations(_FakeImage(), None)  # type: ignore[arg-type]

    def test_rejects_non_int_size(self) -> None:
        with pytest.raises(ValueError, match="size must be an integer"):
            legacy_setup._apply_icon_optimizations(_FakeImage(), 32.0)  # type: ignore[arg-type]

    def test_rejects_non_positive_size(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            legacy_setup._apply_icon_optimizations(_FakeImage(), 0)

    def test_small_icon_uses_sharpen_chain(self) -> None:
        img = _FakeImage()
        enhanced = _FakeImage()
        enhancer = MagicMock()
        enhancer.enhance.return_value = enhanced
        with (
            patch("PIL.ImageEnhance.Contrast", return_value=enhancer) as contrast,
            patch("PIL.ImageFilter.UnsharpMask") as unsharp,
            patch("PIL.ImageFilter.SHARPEN", new="SHARPEN"),
        ):
            result = legacy_setup._apply_icon_optimizations(img, 16)
        contrast.assert_called_once_with(img)
        enhancer.enhance.assert_called_once_with(1.2)
        unsharp.assert_called_once_with(radius=0.5, percent=200, threshold=0)
        # Two filters applied to the enhanced image
        assert len(enhanced.filters) == 2
        assert result is enhanced

    def test_medium_icon_uses_unsharp_only(self) -> None:
        img = _FakeImage()
        with patch("PIL.ImageFilter.UnsharpMask") as unsharp:
            result = legacy_setup._apply_icon_optimizations(img, 48)
        unsharp.assert_called_once_with(radius=0.8, percent=125, threshold=2)
        assert result is img
        assert len(img.filters) == 1

    def test_large_icon_unchanged(self) -> None:
        img = _FakeImage()
        result = legacy_setup._apply_icon_optimizations(img, 256)
        assert result is img
        assert img.filters == []


# ---------------------------------------------------------------------------
# create_optimized_icon
# ---------------------------------------------------------------------------


class TestCreateOptimizedIcon:
    def test_rejects_none_source(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="Source path must not be None"):
            legacy_setup.create_optimized_icon(None, tmp_path / "out.ico")  # type: ignore[arg-type]

    def test_rejects_none_output(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="Output path must not be None"):
            legacy_setup.create_optimized_icon(tmp_path / "x.png", None)  # type: ignore[arg-type]

    def test_rejects_non_path_source(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="source_path must be a Path"):
            legacy_setup.create_optimized_icon("x.png", tmp_path / "o.ico")  # type: ignore[arg-type]

    def test_rejects_non_path_output(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="output_path must be a Path"):
            legacy_setup.create_optimized_icon(tmp_path / "x.png", "o.ico")  # type: ignore[arg-type]

    def test_missing_source_returns_false(self, tmp_path: pathlib.Path) -> None:
        missing = tmp_path / "nope.png"
        assert not legacy_setup.create_optimized_icon(missing, tmp_path / "o.ico")

    def test_successful_icon_generation(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "src.png"
        source.write_bytes(b"not a real png")
        output = tmp_path / "nested" / "out.ico"

        # Construct a fake PIL pipeline.
        saved = MagicMock()
        resized_image = MagicMock()
        resized_image.width = 256
        resized_image.height = 256
        resized_image.save = saved.save

        opened = MagicMock()
        opened.mode = "RGB"  # exercises the convert() branch
        opened.convert.return_value = opened
        opened.resize.return_value = resized_image

        with (
            patch("PIL.Image.open", return_value=opened),
            patch(
                "installer.legacy_setup._apply_icon_optimizations",
                side_effect=lambda img, size: SimpleNamespace(
                    width=size, height=size, save=saved.save
                ),
            ),
        ):
            assert legacy_setup.create_optimized_icon(source, output)
        # save was invoked once on the largest icon
        assert saved.save.called
        assert output.parent.exists()

    def test_exception_returns_false(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "src.png"
        source.write_bytes(b"x")
        with patch("PIL.Image.open", side_effect=RuntimeError("boom")):
            assert not legacy_setup.create_optimized_icon(source, tmp_path / "o.ico")


# ---------------------------------------------------------------------------
# create_shortcut_windows
# ---------------------------------------------------------------------------


class TestCreateShortcutWindows:
    def _kwargs(self, tmp_path: pathlib.Path) -> dict[str, object]:
        return {
            "target_script": "launch.py",
            "working_dir": tmp_path,
            "icon_path": tmp_path / "icon.ico",
            "description": "desc",
        }

    def test_rejects_empty_script(self, tmp_path: pathlib.Path) -> None:
        kw = self._kwargs(tmp_path)
        kw["target_script"] = ""
        with pytest.raises(ValueError, match="Target script must not be empty"):
            legacy_setup.create_shortcut_windows(**kw)  # type: ignore[arg-type]

    def test_rejects_non_string_script(self, tmp_path: pathlib.Path) -> None:
        kw = self._kwargs(tmp_path)
        kw["target_script"] = 123  # truthy non-string
        with pytest.raises(ValueError, match="target_script must be a string"):
            legacy_setup.create_shortcut_windows(**kw)  # type: ignore[arg-type]

    def test_rejects_non_path_working_dir(self, tmp_path: pathlib.Path) -> None:
        kw = self._kwargs(tmp_path)
        kw["working_dir"] = str(tmp_path)
        with pytest.raises(ValueError, match="working_dir must be a Path"):
            legacy_setup.create_shortcut_windows(**kw)  # type: ignore[arg-type]

    def test_rejects_non_path_icon(self, tmp_path: pathlib.Path) -> None:
        kw = self._kwargs(tmp_path)
        kw["icon_path"] = str(tmp_path / "icon.ico")
        with pytest.raises(ValueError, match="icon_path must be a Path"):
            legacy_setup.create_shortcut_windows(**kw)  # type: ignore[arg-type]

    def test_rejects_non_string_description(self, tmp_path: pathlib.Path) -> None:
        kw = self._kwargs(tmp_path)
        kw["description"] = 0xBEEF  # truthy non-string
        with pytest.raises(ValueError, match="description must be a string"):
            legacy_setup.create_shortcut_windows(**kw)  # type: ignore[arg-type]

    def test_success_invokes_powershell(self, tmp_path: pathlib.Path) -> None:
        with patch("installer.legacy_setup.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0)
            assert legacy_setup.create_shortcut_windows(**self._kwargs(tmp_path))
        cmd = run.call_args[0][0]
        assert cmd[0] == "powershell"
        assert "-Command" in cmd

    def test_failure_returns_false_and_logs(self, tmp_path: pathlib.Path) -> None:
        err = subprocess.CalledProcessError(1, "ps", stderr=b"oops")
        with patch("installer.legacy_setup.subprocess.run", side_effect=err):
            assert not legacy_setup.create_shortcut_windows(**self._kwargs(tmp_path))

    def test_failure_without_stderr(self, tmp_path: pathlib.Path) -> None:
        err = subprocess.CalledProcessError(1, "ps", stderr=None)
        with patch("installer.legacy_setup.subprocess.run", side_effect=err):
            assert not legacy_setup.create_shortcut_windows(**self._kwargs(tmp_path))


# ---------------------------------------------------------------------------
# _find_source_image
# ---------------------------------------------------------------------------


class TestFindSourceImage:
    def test_returns_first_existing_candidate(self, tmp_path: pathlib.Path) -> None:
        asset = tmp_path / "src" / "launchers" / "assets"
        asset.mkdir(parents=True)
        chosen = asset / "golf_robot_cropped.png"
        chosen.write_bytes(b"data")
        assert legacy_setup._find_source_image(tmp_path) == chosen

    def test_falls_back_to_repo_root(self, tmp_path: pathlib.Path) -> None:
        fallback = tmp_path / "GolfingRobot.png"
        fallback.write_bytes(b"data")
        assert legacy_setup._find_source_image(tmp_path) == fallback

    def test_returns_none_when_no_candidates(self, tmp_path: pathlib.Path) -> None:
        assert legacy_setup._find_source_image(tmp_path) is None

    def test_prefers_earlier_candidate(self, tmp_path: pathlib.Path) -> None:
        asset = tmp_path / "src" / "launchers" / "assets"
        asset.mkdir(parents=True)
        (asset / "golf_robot_cropped.png").write_bytes(b"a")
        (asset / "golf_robot_icon.png").write_bytes(b"b")
        # cropped takes precedence over icon
        assert (
            legacy_setup._find_source_image(tmp_path).name == "golf_robot_cropped.png"
        )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


@pytest.fixture
def main_env(tmp_path: pathlib.Path):
    """Patch the side-effecting collaborators used by ``main``."""
    with (
        patch("installer.legacy_setup.git_sync_repository") as sync,
        patch("installer.legacy_setup.check_python_dependencies") as deps,
        patch("installer.legacy_setup.get_repo_root", return_value=tmp_path) as root,
        patch("installer.legacy_setup.create_optimized_icon") as gen_icon,
        patch("installer.legacy_setup.create_shortcut_windows") as shortcut,
        patch("installer.legacy_setup.platform.system") as system,
    ):
        deps.return_value = True
        gen_icon.return_value = True
        shortcut.return_value = True
        yield SimpleNamespace(
            sync=sync,
            deps=deps,
            root=root,
            gen_icon=gen_icon,
            shortcut=shortcut,
            system=system,
            tmp_path=tmp_path,
        )


def _seed_source_image(tmp_path: pathlib.Path) -> pathlib.Path:
    asset = tmp_path / "src" / "launchers" / "assets"
    asset.mkdir(parents=True)
    src = asset / "golf_robot_cropped.png"
    src.write_bytes(b"x")
    return src


class TestMain:
    def test_dependency_failure_returns_one(self, main_env) -> None:
        main_env.deps.return_value = False
        assert legacy_setup.main() == 1

    def test_windows_path_success(self, main_env) -> None:
        _seed_source_image(main_env.tmp_path)
        main_env.system.return_value = "Windows"
        assert legacy_setup.main() == 0
        main_env.sync.assert_called_once()
        main_env.shortcut.assert_called_once()

    def test_non_windows_skips_shortcut(self, main_env) -> None:
        _seed_source_image(main_env.tmp_path)
        main_env.system.return_value = "Linux"
        assert legacy_setup.main() == 0
        main_env.shortcut.assert_not_called()

    def test_uses_fallback_when_generation_fails(self, main_env) -> None:
        _seed_source_image(main_env.tmp_path)
        fallback = (
            main_env.tmp_path / "src" / "launchers" / "assets" / "golf_robot_icon.ico"
        )
        fallback.write_bytes(b"ico")
        main_env.gen_icon.return_value = False
        main_env.system.return_value = "Windows"
        assert legacy_setup.main() == 0
        passed_icon = main_env.shortcut.call_args.kwargs["icon_path"]
        assert passed_icon == fallback

    def test_uses_fallback_when_no_source_image(self, main_env) -> None:
        fallback = (
            main_env.tmp_path / "src" / "launchers" / "assets" / "golf_robot_icon.ico"
        )
        fallback.parent.mkdir(parents=True)
        fallback.write_bytes(b"ico")
        main_env.system.return_value = "Windows"
        assert legacy_setup.main() == 0
        # When source_icon is None and fallback exists, shortcut still receives it
        passed_icon = main_env.shortcut.call_args.kwargs["icon_path"]
        assert passed_icon == fallback

    def test_windows_uses_relative_script_path(self, main_env) -> None:
        _seed_source_image(main_env.tmp_path)
        main_env.system.return_value = "Windows"
        # Create the launch script so relative_to succeeds
        (main_env.tmp_path / "launch_golf_suite.py").write_text("")
        assert legacy_setup.main() == 0
        target = main_env.shortcut.call_args.kwargs["target_script"]
        assert target == "launch_golf_suite.py"

    def test_windows_handles_absolute_script_path_value_error(
        self, main_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ValueError branch when ``relative_to`` cannot succeed."""
        _seed_source_image(main_env.tmp_path)
        main_env.system.return_value = "Windows"
        # Force relative_to to fail by changing get_repo_root after path resolution.
        # Replace the launch script's relative_to with one that raises.
        original_path_cls = pathlib.Path

        class _AbsolutePath(original_path_cls):  # type: ignore[misc]
            def relative_to(self, *args: object, **kwargs: object):
                raise ValueError("not relative")

        # Patch only the launch script creation by monkeypatching get_repo_root to
        # return a path object whose joined launch script returns an _AbsolutePath.
        fake_root = _AbsolutePath(main_env.tmp_path)
        monkeypatch.setattr(legacy_setup, "get_repo_root", lambda: fake_root)
        assert legacy_setup.main() == 0
        target = main_env.shortcut.call_args.kwargs["target_script"]
        # Absolute path returned as string when relative_to fails
        assert "launch_golf_suite.py" in target
