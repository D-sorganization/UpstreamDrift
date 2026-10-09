"""Source execution receipts and saved-bundle integrity contracts."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.tools.shot_pattern_analysis.provenance import (
    assert_source_unchanged,
    manifest_files_match,
)

pytestmark = pytest.mark.unit


def test_changed_source_is_rejected_after_an_analysis() -> None:
    before = {"source_sha256": {"core.py": "abc"}, "native_binary_sha256": "123"}
    assert_source_unchanged(before, before)
    with pytest.raises(RuntimeError, match="source changed"):
        assert_source_unchanged(before, {**before, "source_sha256": {"core.py": "def"}})
    with pytest.raises(RuntimeError, match="source changed"):
        assert_source_unchanged(before, {**before, "native_binary_sha256": "456"})


def test_complete_manifest_requires_matching_file_hashes(tmp_path) -> None:
    payload = b"shot data"
    (tmp_path / "shots.csv").write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files_sha256": {"shots.csv": hashlib.sha256(payload).hexdigest()}})
    )
    assert manifest_files_match(manifest)
    (tmp_path / "shots.csv").write_bytes(b"different")
    assert not manifest_files_match(manifest)
    (tmp_path / "shots.csv").unlink()
    assert not manifest_files_match(manifest)
