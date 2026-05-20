"""Branch-coverage backfill for installer modules (wave 9)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from installer.windows import build_installer, packaging_profiles
from installer.windows.packaging_profiles import PackagingProfile


# ---------------------------------------------------------------------------
# PackagingProfile validation
# ---------------------------------------------------------------------------


def test_packaging_profile_rejects_invalid_discovery_mode() -> None:
    with pytest.raises(ValueError, match="invalid discovery mode"):
        PackagingProfile(
            profile_id="x",
            display_name="X",
            description="d",
            discovery_mode="bogus",
            include_api_executable=False,
            bundle_optional_engines=False,
        )


def test_get_packaging_profile_normalizes_case_and_whitespace() -> None:
    profile = packaging_profiles.get_packaging_profile("  CORE  ")
    assert profile.profile_id == "core"


# ---------------------------------------------------------------------------
# build_installer.check_prerequisites — virtualenv branch
# ---------------------------------------------------------------------------


def test_check_prerequisites_succeeds_with_cx_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cx = SimpleNamespace(version="6.15.0")
    monkeypatch.setitem(sys.modules, "cx_Freeze", fake_cx)
    assert build_installer.check_prerequisites() is True


def test_check_prerequisites_handles_missing_cx_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "cx_Freeze", raising=False)
    # Force ImportError by inserting a finder that rejects cx_Freeze
    original_import = build_installer.__builtins__["__import__"]  # type: ignore[index]

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "cx_Freeze":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setitem(
        build_installer.__builtins__,  # type: ignore[index,arg-type]
        "__import__",
        fake_import,
    )
    assert build_installer.check_prerequisites() is False


# ---------------------------------------------------------------------------
# build_installer.main — early-exit branches
# ---------------------------------------------------------------------------


@pytest.fixture
def main_patches():
    """Patch all collaborators of ``build_installer.main``."""
    argv = ["build_installer"]
    with (
        patch.object(sys, "argv", argv),
        patch("installer.windows.build_installer.check_prerequisites") as prereq,
        patch("installer.windows.build_installer.clean_build_dirs") as clean,
        patch("installer.windows.build_installer.install_dependencies") as deps,
        patch("installer.windows.build_installer.detect_physics_engines") as engines,
        patch("installer.windows.build_installer.build_executable") as build_exe,
        patch("installer.windows.build_installer.build_msi") as build_msi,
        patch("installer.windows.build_installer.create_installer_info") as info,
        patch("installer.windows.build_installer._log_generated_outputs") as logger,
        patch("installer.windows.build_installer.DIST_DIR") as dist,
    ):
        prereq.return_value = True
        deps.return_value = True
        engines.return_value = ["mujoco"]
        build_exe.return_value = True
        build_msi.return_value = True
        dist.glob.return_value = []
        yield SimpleNamespace(
            argv=argv,
            prereq=prereq,
            clean=clean,
            deps=deps,
            engines=engines,
            build_exe=build_exe,
            build_msi=build_msi,
            info=info,
            logger=logger,
            dist=dist,
        )


def test_main_exits_when_prerequisites_fail(main_patches) -> None:
    main_patches.prereq.return_value = False
    with pytest.raises(SystemExit) as exc:
        build_installer.main()
    assert exc.value.code == 1


def test_main_exits_when_dependency_install_fails(main_patches) -> None:
    main_patches.deps.return_value = False
    with pytest.raises(SystemExit) as exc:
        build_installer.main()
    assert exc.value.code == 1


def test_main_skip_deps_bypasses_install(main_patches) -> None:
    main_patches.argv.extend(["--skip-deps", "--exe-only"])
    build_installer.main()
    main_patches.deps.assert_not_called()


def test_main_exits_when_no_engines_detected(main_patches) -> None:
    main_patches.engines.return_value = []
    with pytest.raises(SystemExit) as exc:
        build_installer.main()
    assert exc.value.code == 1


def test_main_exits_when_build_executable_fails(main_patches) -> None:
    main_patches.build_exe.return_value = False
    with pytest.raises(SystemExit) as exc:
        build_installer.main()
    assert exc.value.code == 1


def test_main_exits_when_build_msi_fails(main_patches) -> None:
    main_patches.build_msi.return_value = False
    with pytest.raises(SystemExit) as exc:
        build_installer.main()
    assert exc.value.code == 1


def test_main_clean_invokes_clean_build_dirs(main_patches) -> None:
    main_patches.argv.extend(["--clean", "--exe-only", "--skip-deps"])
    build_installer.main()
    main_patches.clean.assert_called_once()


def test_main_logs_generated_outputs_when_present(main_patches, tmp_path) -> None:
    main_patches.argv.extend(["--exe-only", "--skip-deps"])
    sentinel = tmp_path / "out.msi"
    sentinel.write_bytes(b"x")
    main_patches.dist.glob.return_value = [sentinel]
    build_installer.main()
    main_patches.logger.assert_called_once_with([sentinel])


# ---------------------------------------------------------------------------
# detect_physics_engines — uses real importer
# ---------------------------------------------------------------------------


def test_detect_physics_engines_returns_list() -> None:
    result = build_installer.detect_physics_engines()
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)
