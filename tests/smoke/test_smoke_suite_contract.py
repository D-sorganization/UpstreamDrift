"""Contract tests for release-blocking artifact smoke suites."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SMOKE_ROOT = REPO_ROOT / "tests" / "smoke"
CANONICAL_SMOKE_SUITES = {
    "python_wheel",
    "docker_api",
    "tauri_desktop",
    "rust_crate",
}

pytestmark = pytest.mark.smoke


def test_each_canonical_artifact_has_smoke_suite() -> None:
    """Every canonical artifact must have a release smoke-test directory."""
    discovered = {
        path.name
        for path in SMOKE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }

    assert discovered == CANONICAL_SMOKE_SUITES


def test_each_smoke_suite_has_pytest_configuration() -> None:
    """Suites must be runnable in isolation with local pytest markers/options."""
    for suite_name in CANONICAL_SMOKE_SUITES:
        assert (SMOKE_ROOT / suite_name / "pytest.ini").is_file()


def test_each_smoke_suite_has_artifact_test() -> None:
    """Suites must contain artifact-facing tests, not only package metadata tests."""
    for suite_name in CANONICAL_SMOKE_SUITES:
        suite = SMOKE_ROOT / suite_name
        test_files = {test_file.name for test_file in suite.glob("test_*.py")}

        assert test_files, f"{suite_name} has no pytest files"
        assert any(
            "artifact" in name or "image" in name or "bundle" in name
            for name in test_files
        )
