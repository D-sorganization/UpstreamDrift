"""Tests for the reviewed-test-deletion gate (#9412).

The workflow guard added by #7368 refuses any deletion under ``tests/``. This
script is the mechanism that lets a deletion be *recorded* as reviewed without
weakening the guard: an unrecorded deletion still fails.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from importlib import util
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ci" / "check_reviewed_test_deletions.py"
_MANIFEST_PATH = _REPO_ROOT / "scripts" / "config" / "reviewed_test_deletions.json"


def _load_module() -> Any:
    """Import the gate script by path, as CI invokes it."""
    spec = util.spec_from_file_location(
        "reviewed_test_deletions_under_test", _SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load reviewed-test-deletion script")
    module = util.module_from_spec(spec)
    # `dataclasses` resolves field types through `sys.modules[cls.__module__]`,
    # so the module must be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> Any:
    """The loaded gate module."""
    return _load_module()


def _write_manifest(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    """Write a manifest containing ``entries`` and return its path."""
    path = tmp_path / "reviewed_test_deletions.json"
    path.write_text(
        json.dumps({"schema_version": 1, "reviewed_deletions": entries}),
        encoding="utf-8",
    )
    return path


def _entry(**overrides: Any) -> dict[str, Any]:
    """A valid manifest entry, with overrides applied."""
    entry: dict[str, Any] = {
        "path": "tests/config/test_gone.py",
        "replacement": "tests/config/test_successor.py",
        "reason": "superseded",
        "tracked_in": "https://example.invalid/issues/1",
        "expires_on": "2099-01-01",
    }
    entry.update(overrides)
    return entry


class TestGateBehaviour:
    """The gate accepts recorded deletions and rejects everything else."""

    def test_unrecorded_deletion_fails(self, gate: Any, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, [])
        listing = tmp_path / "deleted.txt"
        listing.write_text("tests/config/test_gone.py\n", encoding="utf-8")
        exit_code = gate.main(
            [
                "--deleted-files",
                str(listing),
                "--config",
                str(manifest),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert exit_code == 1

    def test_recorded_deletion_passes(self, gate: Any, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, [_entry()])
        (tmp_path / "tests" / "config").mkdir(parents=True)
        (tmp_path / "tests" / "config" / "test_successor.py").write_text(
            "", encoding="utf-8"
        )
        exit_code = gate.main(
            [
                "tests/config/test_gone.py",
                "--config",
                str(manifest),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert exit_code == 0

    def test_missing_replacement_fails(self, gate: Any, tmp_path: Path) -> None:
        """A replacement that does not exist is an unchecked claim."""
        manifest = _write_manifest(tmp_path, [_entry()])
        exit_code = gate.main(
            [
                "tests/config/test_gone.py",
                "--config",
                str(manifest),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert exit_code == 1

    def test_expired_record_fails(self, gate: Any, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, [_entry(expires_on="2000-01-01")])
        exit_code = gate.main(
            [
                "tests/config/test_gone.py",
                "--config",
                str(manifest),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert exit_code == 1

    def test_retired_test_needs_no_replacement(self, gate: Any, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, [_entry(replacement=None)])
        exit_code = gate.main(
            [
                "tests/config/test_gone.py",
                "--config",
                str(manifest),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert exit_code == 0

    def test_no_deletions_is_a_no_op(self, gate: Any, tmp_path: Path) -> None:
        assert gate.main(["--config", str(tmp_path / "absent.json")]) == 0

    def test_entry_without_reason_is_rejected(self, gate: Any, tmp_path: Path) -> None:
        manifest = _write_manifest(tmp_path, [_entry(reason="  ")])
        exit_code = gate.main(
            [
                "tests/config/test_gone.py",
                "--config",
                str(manifest),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert exit_code == 1


class TestShippedManifest:
    """The manifest committed to this repository must itself be valid."""

    def test_manifest_parses_and_every_replacement_exists(self, gate: Any) -> None:
        entries = gate.load_entries(_MANIFEST_PATH)
        assert entries, "manifest should record at least one reviewed deletion"
        for path, entry in entries.items():
            assert not (_REPO_ROOT / path).exists(), (
                f"{path} is recorded as deleted but still exists"
            )
            assert entry.expires_on > date.today()
            if entry.replacement is not None:
                assert (_REPO_ROOT / entry.replacement).is_file()
