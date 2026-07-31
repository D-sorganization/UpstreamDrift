"""T6 TDD: console-script entry point and packaging metadata are correct."""

from __future__ import annotations

import os
import subprocess  # nosec B404 - fixed interpreter and local import probes
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent
pytestmark = pytest.mark.unit

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _tools_root() -> Path:
    return Path(os.environ.get("TOOLS_REPO_PATH", ROOT / "vendor/ud-tools")).resolve()


def _canonical_environment() -> dict[str, str]:
    tools_root = _tools_root()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(tools_root / "src/shared/python"),
            str(tools_root / "src"),
            str(tools_root / "src/python/src"),
        )
    )
    return env


# ---------------------------------------------------------------------------
# T6-AC-1: console-script entry point is declared correctly
# ---------------------------------------------------------------------------


def test_sidekick_script_declared(pyproject: dict) -> None:
    scripts = pyproject.get("project", {}).get("scripts", {})
    assert (
        "sidekick" in scripts
    ), "sidekick console script missing from [project.scripts]"


def test_sidekick_script_points_to_main(pyproject: dict) -> None:
    scripts = pyproject["project"]["scripts"]
    assert (
        scripts["sidekick"] == "sidekick.__main__:main"
    ), f"Expected 'sidekick.__main__:main', got '{scripts['sidekick']}'"


# ---------------------------------------------------------------------------
# T6-AC-2: entry point module + callable are importable (no GUI required)
# ---------------------------------------------------------------------------


def test_sidekick_main_importable() -> None:
    script = """
from pathlib import Path
import os
from sidekick import __main__
assert callable(__main__.main)
assert Path(__main__.__file__).resolve().is_relative_to(
    Path(os.environ["TOOLS_REPO_PATH"]).resolve()
)
"""
    env = _canonical_environment()
    env["TOOLS_REPO_PATH"] = str(_tools_root())
    result = subprocess.run(  # nosec B603 - fixed interpreter and inline probe
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# T6-AC-3: wheel excludes docs/, tests/, vendor/
# ---------------------------------------------------------------------------


def test_hatch_build_includes_src(pyproject: dict) -> None:
    includes = (
        pyproject.get("tool", {}).get("hatch", {}).get("build", {}).get("include", [])
    )
    assert any("src/**" in p for p in includes), "hatch build.include must cover src/**"


def test_hatch_wheel_delegates_tools_packages_to_custom_hook(
    pyproject: dict,
) -> None:
    wheel_target = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    force_include = wheel_target.get("force-include", {})
    assert "src/shared/python/sidekick" not in force_include
    assert "src/shared/python/chat" not in force_include
    custom_hook = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("hooks", {})
        .get("custom", {})
    )
    assert custom_hook.get("path") == "build_hooks.py"


def test_hatch_build_excludes_docs(pyproject: dict) -> None:
    """docs/ and tests/ should NOT appear as explicit includes."""
    includes = (
        pyproject.get("tool", {}).get("hatch", {}).get("build", {}).get("include", [])
    )
    for path in includes:
        assert (
            "docs/" not in path
        ), f"docs/ should not be an explicit wheel include: {path}"
        assert (
            "tests/" not in path
        ), f"tests/ should not be an explicit wheel include: {path}"
        assert (
            "vendor/" not in path
        ), f"vendor/ should not be an explicit wheel include: {path}"


# ---------------------------------------------------------------------------
# T6-AC-4: sidekick.standalone package exists and has __init__.py
# ---------------------------------------------------------------------------


def test_standalone_is_parent_owned_without_downstream_copy() -> None:
    relative_sources = (
        "sidekick/__main__.py",
        "sidekick/persistence/__init__.py",
        "sidekick/persistence/schema.py",
        "sidekick/persistence/state_profile.py",
        "sidekick/standalone/__init__.py",
        "sidekick/standalone/onboarding.py",
        "sidekick/standalone/preferences.py",
        "sidekick/standalone/runner.py",
        "sidekick/standalone/session_store.py",
        "sidekick/standalone/window.py",
    )
    local_python = ROOT / "src/shared/python"
    parent_python = _tools_root() / "src/shared/python"
    for relative in relative_sources:
        assert not (local_python / relative).exists()
        assert (parent_python / relative).is_file()


# ---------------------------------------------------------------------------
# T6-AC-5: sidekick --help exits 0 (subprocess test for installed behaviour)
# ---------------------------------------------------------------------------


def test_sidekick_help_argparse() -> None:
    """The canonical parent CLI must execute headlessly from product source."""
    result = subprocess.run(  # nosec B603 - fixed interpreter and module
        [sys.executable, "-m", "sidekick", "--help"],
        cwd=ROOT,
        env=_canonical_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Standalone Sidekick launcher and headless dispatcher." in result.stdout
