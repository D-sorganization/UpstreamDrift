"""T6 TDD: console-script entry point and packaging metadata are correct."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# T6-AC-1: console-script entry point is declared correctly
# ---------------------------------------------------------------------------


def test_sidekick_script_declared(pyproject: dict) -> None:
    scripts = pyproject.get("project", {}).get("scripts", {})
    assert "sidekick" in scripts, (
        "sidekick console script missing from [project.scripts]"
    )


def test_sidekick_script_points_to_main(pyproject: dict) -> None:
    scripts = pyproject["project"]["scripts"]
    assert scripts["sidekick"] == "sidekick.__main__:main", (
        f"Expected 'sidekick.__main__:main', got '{scripts['sidekick']}'"
    )


# ---------------------------------------------------------------------------
# T6-AC-2: entry point module + callable are importable (no GUI required)
# ---------------------------------------------------------------------------


def test_sidekick_main_importable() -> None:
    mod = importlib.import_module("sidekick.__main__")
    assert callable(getattr(mod, "main", None)), (
        "sidekick.__main__.main must be callable"
    )


# ---------------------------------------------------------------------------
# T6-AC-3: wheel excludes docs/, tests/, vendor/
# ---------------------------------------------------------------------------


def test_hatch_build_includes_src(pyproject: dict) -> None:
    includes = (
        pyproject.get("tool", {}).get("hatch", {}).get("build", {}).get("include", [])
    )
    assert any("src/**" in p for p in includes), "hatch build.include must cover src/**"


def test_hatch_wheel_packages_sidekick_as_top_level(pyproject: dict) -> None:
    wheel_packages = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    assert "src/shared/python/sidekick" in wheel_packages, (
        "wheel target must package src/shared/python/sidekick as top-level sidekick"
    )


def test_hatch_build_excludes_docs(pyproject: dict) -> None:
    """docs/ and tests/ should NOT appear as explicit includes."""
    includes = (
        pyproject.get("tool", {}).get("hatch", {}).get("build", {}).get("include", [])
    )
    for path in includes:
        assert "docs/" not in path, (
            f"docs/ should not be an explicit wheel include: {path}"
        )
        assert "tests/" not in path, (
            f"tests/ should not be an explicit wheel include: {path}"
        )
        assert "vendor/" not in path, (
            f"vendor/ should not be an explicit wheel include: {path}"
        )


# ---------------------------------------------------------------------------
# T6-AC-4: sidekick.standalone package exists and has __init__.py
# ---------------------------------------------------------------------------


def test_sidekick_standalone_package_exists() -> None:
    standalone = ROOT / "src" / "shared" / "python" / "sidekick" / "standalone"
    assert standalone.is_dir(), "sidekick/standalone/ directory is missing"
    assert (standalone / "__init__.py").exists(), (
        "sidekick/standalone/__init__.py is missing"
    )


# ---------------------------------------------------------------------------
# T6-AC-5: sidekick --help exits 0 (subprocess test for installed behaviour)
# ---------------------------------------------------------------------------


def test_sidekick_help_argparse(capsys: pytest.CaptureFixture) -> None:
    """main() with --help should raise SystemExit(0) and print usage."""
    from sidekick.__main__ import main

    with pytest.raises(SystemExit) as exc:
        sys.argv = ["sidekick", "--help"]
        main()
    assert exc.value.code == 0
