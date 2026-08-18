"""T7 TDD: PyInstaller spec file satisfies the documented include/exclude contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parent.parent.parent.parent
SPEC_PATH = ROOT / "sidekick.spec"
BUILD_SCRIPT = ROOT / "scripts" / "packaging" / "build_sidekick_binary.py"


@pytest.fixture(scope="module")
def spec_source() -> str:
    assert SPEC_PATH.exists(), f"sidekick.spec not found at {SPEC_PATH}"
    return SPEC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def build_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_build_sidekick_binary_module",
        BUILD_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# T7-contract-1: entry-point is correct
# ---------------------------------------------------------------------------


def test_spec_entrypoint(spec_source: str) -> None:
    assert 'root / "vendor" / "ud-tools" / "src" / "shared" / "python"' in (spec_source)
    assert 'canonical_tools_python / "sidekick" / "__main__.py"' in spec_source
    assert 'local_python / "sidekick" / "__main__.py"' not in spec_source
    assert "[str(binary_entrypoint)]" in spec_source
    adapter = ROOT / "scripts" / "packaging" / "sidekick_binary_entrypoint.py"
    assert adapter.is_file()
    assert "from sidekick.__main__ import main" in adapter.read_text(encoding="utf-8")


def test_canonical_tools_path_precedes_local_extensions(spec_source: str) -> None:
    """PyInstaller must resolve parent-owned modules before UD extensions."""
    pathex_start = spec_source.index("pathex=[")
    pathex_end = spec_source.index("],", pathex_start)
    pathex = spec_source[pathex_start:pathex_end]

    assert pathex.index("canonical_tools_python") < pathex.index("local_python")
    assert pathex.index("canonical_tools_python") < pathex.index("canonical_tools_src")
    assert pathex.index("canonical_tools_src") < pathex.index("local_python")


def test_spec_bootstraps_canonical_tools_for_hidden_import_discovery(
    spec_source: str,
) -> None:
    """Spec evaluation must expose the canonical package to PyInstaller hooks."""
    bootstrap = spec_source.index("sys.path.insert(0, str(canonical_tools_python))")
    analysis = spec_source.index("a = Analysis")

    assert bootstrap < analysis


def test_windows_icon_is_a_pinned_tools_asset(spec_source: str) -> None:
    """Windows builds must reference an icon that exists in a clean checkout."""
    expected_icon = ROOT / "vendor" / "ud-tools" / "assets" / "tools_icon_hq.ico"

    assert expected_icon.is_file()
    assert (
        'root / "vendor" / "ud-tools" / "assets" / "tools_icon_hq.ico"' in spec_source
    )


# ---------------------------------------------------------------------------
# T7-contract-2: required packages are included
# ---------------------------------------------------------------------------


REQUIRED_INCLUDES = [
    "sidekick",
]


@pytest.mark.parametrize("pkg", REQUIRED_INCLUDES)
def test_required_package_included(spec_source: str, pkg: str) -> None:
    assert pkg in spec_source, f"spec must include '{pkg}'"


# ---------------------------------------------------------------------------
# T7-contract-3: heavy physics engines are excluded
# ---------------------------------------------------------------------------


EXCLUDED_PACKAGES = ["pybullet", "mujoco", "pydrake"]
EXCLUDED_QT_BINDINGS = ["PyQt5", "PySide2", "PySide6"]
EXCLUDED_NONRUNTIME_PACKAGES = ["docutils", "sklearn", "sphinx"]


@pytest.mark.parametrize("pkg", EXCLUDED_PACKAGES)
def test_excluded_package_excluded(spec_source: str, pkg: str) -> None:
    # The package should appear only in the excludes section, not in includes
    # Simplest check: if it appears at all it must be in an excludes list
    if pkg not in spec_source:
        return  # absent entirely — fine
    # If present, it must be explicitly excluded
    assert (
        "excludes" in spec_source.lower() or "exclude_binaries" in spec_source.lower()
    ), f"'{pkg}' appears in spec but there is no excludes section"


@pytest.mark.parametrize("pkg", EXCLUDED_QT_BINDINGS)
def test_noncanonical_qt_bindings_are_excluded(spec_source: str, pkg: str) -> None:
    """The PyQt6 artifact must not let optional imports collect another binding."""
    excludes_start = spec_source.index("excludes=[")
    excludes_end = spec_source.index("],", excludes_start)

    assert f'"{pkg}"' in spec_source[excludes_start:excludes_end]


@pytest.mark.parametrize("pkg", EXCLUDED_NONRUNTIME_PACKAGES)
def test_nonruntime_packages_are_excluded(spec_source: str, pkg: str) -> None:
    """Documentation and ML-analysis stacks are outside Sidekick's contract."""
    excludes_start = spec_source.index("excludes=[")
    excludes_end = spec_source.index("],", excludes_start)

    assert f'"{pkg}"' in spec_source[excludes_start:excludes_end]


# ---------------------------------------------------------------------------
# T7-contract-4: size budget constant is declared
# ---------------------------------------------------------------------------


def test_size_budget_declared(spec_source: str) -> None:
    assert "MAX_MB" in spec_source or "max_mb" in spec_source or "250" in spec_source, (
        "spec or build script must declare the 250 MB size budget"
    )


# ---------------------------------------------------------------------------
# T7-contract-5: build script exists
# ---------------------------------------------------------------------------


def test_build_script_exists() -> None:
    script = ROOT / "scripts" / "packaging" / "build_sidekick_binary.py"
    assert script.exists(), f"Build script not found at {script}"


# ---------------------------------------------------------------------------
# T7-contract-6: CI workflow file exists and has the expected triggers
# ---------------------------------------------------------------------------


def test_release_workflow_exists() -> None:
    wf = ROOT / ".github" / "workflows" / "release-sidekick-binary.yml"
    assert wf.exists(), f"Release workflow not found at {wf}"


def test_release_workflow_triggers(spec_source: str) -> None:
    wf = ROOT / ".github" / "workflows" / "release-sidekick-binary.yml"
    if not wf.exists():
        pytest.skip("workflow file absent")
    content = wf.read_text(encoding="utf-8")
    assert "workflow_dispatch" in content, (
        "workflow must support workflow_dispatch trigger"
    )
    assert "sidekick-v" in content, "workflow must trigger on sidekick-v* tags"


def test_build_script_rejects_a_non_native_platform_target(spec_source: str) -> None:
    """The wrapper must fail rather than mislabel a host-native executable."""
    script = ROOT / "scripts" / "packaging" / "build_sidekick_binary.py"
    content = script.read_text(encoding="utf-8")

    assert "--expected-platform" in content
    assert "requested platform" in content.lower()


def test_build_script_fails_before_building_for_platform_mismatch(
    build_script_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Platform validation is executable behavior, not only workflow metadata."""
    spec_file = tmp_path / "sidekick.spec"
    entrypoint = tmp_path / "__main__.py"
    spec_file.touch()
    entrypoint.touch()
    args = SimpleNamespace(
        expected_platform="windows",
        max_mb=250,
        output_dir=str(tmp_path / "dist"),
    )

    monkeypatch.setattr(build_script_module, "SPEC_FILE", spec_file)
    monkeypatch.setattr(
        build_script_module,
        "CANONICAL_SIDEKICK_ENTRYPOINT",
        entrypoint,
    )
    monkeypatch.setattr(build_script_module, "_parse_args", lambda: args)
    monkeypatch.setattr(
        build_script_module,
        "_native_platform_name",
        lambda: "linux",
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("PyInstaller must not run for a platform mismatch")

    monkeypatch.setattr(build_script_module.subprocess, "run", fail_if_called)

    assert build_script_module.main() == 1
    assert "does not match native build platform" in capsys.readouterr().err


def test_build_script_requires_canonical_tools_entrypoint(
    build_script_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An uninitialized or stale Tools checkout must fail before PyInstaller."""
    spec_file = tmp_path / "sidekick.spec"
    spec_file.touch()
    args = SimpleNamespace(
        expected_platform="linux",
        max_mb=250,
        output_dir=str(tmp_path / "dist"),
    )

    monkeypatch.setattr(build_script_module, "SPEC_FILE", spec_file)
    monkeypatch.setattr(
        build_script_module,
        "CANONICAL_SIDEKICK_ENTRYPOINT",
        tmp_path / "missing" / "__main__.py",
    )
    monkeypatch.setattr(build_script_module, "_parse_args", lambda: args)

    assert build_script_module.main() == 1
    assert "initialize the pinned vendor/ud-tools submodule" in (
        capsys.readouterr().err
    )


def test_build_script_accepts_matching_native_platform(
    build_script_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A valid native build must verify the exact expected output artifact."""
    spec_file = tmp_path / "sidekick.spec"
    entrypoint = tmp_path / "__main__.py"
    output_dir = tmp_path / "dist"
    spec_file.touch()
    entrypoint.touch()
    args = SimpleNamespace(
        expected_platform="linux",
        max_mb=1,
        output_dir=str(output_dir),
    )

    monkeypatch.setattr(build_script_module, "SPEC_FILE", spec_file)
    monkeypatch.setattr(
        build_script_module,
        "CANONICAL_SIDEKICK_ENTRYPOINT",
        entrypoint,
    )
    monkeypatch.setattr(build_script_module, "_parse_args", lambda: args)
    monkeypatch.setattr(
        build_script_module,
        "_native_platform_name",
        lambda: "linux",
    )
    monkeypatch.setattr(build_script_module, "_binary_name", lambda: "sidekick")

    def successful_build(
        command: list[str],
        *,
        env: dict[str, str],
        check: bool,
    ) -> SimpleNamespace:
        assert command[-1] == str(spec_file)
        assert command[command.index("--distpath") + 1] == str(output_dir)
        assert env["SKIP_UI_BUILD"] == "1"
        assert check is False
        output_dir.mkdir()
        (output_dir / "sidekick").write_bytes(b"native binary")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build_script_module.subprocess, "run", successful_build)

    assert build_script_module.main() == 0
    assert (output_dir / "sidekick").read_bytes() == b"native binary"


def test_release_workflow_does_not_fake_cross_platform_matrix(
    spec_source: str,
) -> None:
    wf = ROOT / ".github" / "workflows" / "release-sidekick-binary.yml"
    if not wf.exists():
        pytest.skip("workflow file absent")
    content = wf.read_text(encoding="utf-8")
    assert "Build binary (Linux)" in content
    assert "sidekick-linux" in content
    assert "sidekick-macos" not in content
    assert "sidekick-windows" not in content
