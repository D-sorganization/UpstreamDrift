"""Tests for the Docker-safe pinned Tools source attestation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.packaging.pinned_tools_provenance import (
    REQUIRED_TOOLS_SOURCE_PATHS,
    compute_tools_source_sha256,
)

pytestmark = pytest.mark.unit


def _write_required_tree(root: Path) -> None:
    for relative in REQUIRED_TOOLS_SOURCE_PATHS:
        path = root / relative
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"source:{relative.as_posix()}\n", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "__init__.py").write_text(
                f"# {relative.as_posix()}\n",
                encoding="utf-8",
            )


def test_tools_source_digest_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    tools_root = tmp_path / "vendor" / "ud-tools"
    _write_required_tree(tools_root)

    initial = compute_tools_source_sha256(tools_root)
    repeated = compute_tools_source_sha256(tools_root)
    changed_file = tools_root / "src" / "sidekick" / "__init__.py"
    changed_file.write_text("# changed\n", encoding="utf-8")
    changed = compute_tools_source_sha256(tools_root)

    assert len(initial) == 64
    assert initial == repeated
    assert changed != initial


def test_tools_source_digest_fails_closed_when_a_required_root_is_missing(
    tmp_path: Path,
) -> None:
    tools_root = tmp_path / "vendor" / "ud-tools"
    _write_required_tree(tools_root)
    missing = tools_root / "src" / "contracts.py"
    missing.unlink()

    with pytest.raises(ValueError, match="required Tools source roots are missing"):
        compute_tools_source_sha256(tools_root)
