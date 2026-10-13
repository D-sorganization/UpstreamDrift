"""Historical audit sources remain recoverable after implementation cleanup."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "docs/research/shot_pattern_analysis/phase2/validation/refinement_v3/historical_source_snapshots/manifest.json"
)


def test_all_fifteen_historical_source_snapshots_preserve_original_bytes():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["schema"] == "shot-historical-source-snapshots/1"
    assert len(manifest["sources"]) == 15
    assert len({entry["source_path"] for entry in manifest["sources"]}) == 15
    assert len(manifest["source_commit"]) == 40
    for entry in manifest["sources"]:
        archive = (MANIFEST.parent / entry["archive"]).resolve()
        assert archive.is_relative_to(MANIFEST.parent.resolve())
        stored = archive.read_bytes()
        assert hashlib.sha256(stored).hexdigest() == entry["storage_sha256"]
        assert len(stored) == entry["storage_bytes"]
        assert stored[4:8] == b"\0" * 4  # Deterministic gzip timestamp.
        original = gzip.decompress(stored)
        assert len(original) == entry["original_bytes"]
        assert hashlib.sha256(original).hexdigest() == entry["original_sha256"]
        assert (ROOT / entry["source_path"]).is_file()
