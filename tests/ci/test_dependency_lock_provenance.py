"""Verify dependency lock provenance is invariant to environment flags like --no-index."""

from __future__ import annotations

from pathlib import Path
import re
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def test_requirements_lock_provenance_header_is_canonical() -> None:
    """requirements.lock must have a canonical, environment-invariant compile command."""
    lock_path = ROOT / "requirements.lock"
    content = lock_path.read_text(encoding="utf-8")
    header_match = re.search(r"#\s+pip-compile\s+(.*)", content)
    assert header_match is not None, (
        "requirements.lock must contain pip-compile command header"
    )
    cmd = header_match.group(1).strip()
    assert "--no-index" not in cmd, (
        "requirements.lock header must not contain environment-derived --no-index flag"
    )
    assert cmd == "--output-file=requirements.lock pyproject.toml"


def test_requirements_dev_lock_provenance_header_is_canonical() -> None:
    """requirements-dev.lock must have a canonical, environment-invariant compile command."""
    lock_path = ROOT / "requirements-dev.lock"
    content = lock_path.read_text(encoding="utf-8")
    header_match = re.search(r"#\s+pip-compile\s+(.*)", content)
    assert header_match is not None, (
        "requirements-dev.lock must contain pip-compile command header"
    )
    cmd = header_match.group(1).strip()
    assert "--no-index" not in cmd, (
        "requirements-dev.lock header must not contain environment-derived --no-index flag"
    )
    assert (
        cmd
        == "--extra=dev --extra=gui-test --output-file=requirements-dev.lock pyproject.toml"
    )


def test_makefile_sync_deps_uses_custom_compile_command() -> None:
    """Makefile sync-deps target must pass --custom-compile-command for both lockfiles."""
    makefile_path = ROOT / "Makefile"
    content = makefile_path.read_text(encoding="utf-8")
    assert (
        '--custom-compile-command="pip-compile --output-file=requirements.lock pyproject.toml"'
        in content
    )
    assert (
        '--custom-compile-command="pip-compile --extra=dev --extra=gui-test --output-file=requirements-dev.lock pyproject.toml"'
        in content
    )
