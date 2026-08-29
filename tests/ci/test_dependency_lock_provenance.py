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
    """Makefile sync-deps must pin CUSTOM_COMPILE_COMMAND for both lockfiles.

    pip-tools has no ``--custom-compile-command`` CLI flag; the header override is
    supplied through the ``CUSTOM_COMPILE_COMMAND`` environment variable. Pinning it
    is what keeps the generated header free of environment-derived flags such as
    ``--no-index``, so ``make sync-deps`` is reproducible offline.
    """
    makefile_path = ROOT / "Makefile"
    content = makefile_path.read_text(encoding="utf-8")
    assert (
        'CUSTOM_COMPILE_COMMAND="pip-compile --output-file=requirements.lock pyproject.toml"'
        in content
    )
    assert (
        'CUSTOM_COMPILE_COMMAND="pip-compile --extra=dev --extra=gui-test --output-file=requirements-dev.lock pyproject.toml"'
        in content
    )
