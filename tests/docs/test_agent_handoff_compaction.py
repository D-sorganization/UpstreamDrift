"""Integrity and navigation checks for byte-preserving handoff compaction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/agents/handoffs/shot-pattern-handoff-archive-manifest.json"
MAX_CHUNK_BYTES = 51_200
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_repo_path(relative_path: str) -> bytes:
    path = (ROOT / relative_path).resolve()
    assert path.is_relative_to(ROOT.resolve()), (
        f"path escapes repository: {relative_path}"
    )
    return path.read_bytes()


def test_original_handoff_snapshots_match_their_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for snapshot in manifest["snapshots"]:
        payload = _read_repo_path(snapshot["archive_path"])
        assert len(payload) == snapshot["size_bytes"]
        assert _sha256(payload) == snapshot["sha256"]

    for legacy_archive in manifest["legacy_archives"]:
        assert (
            _sha256(_read_repo_path(legacy_archive["path"])) == legacy_archive["sha256"]
        )


def test_peer_record_chunks_reassemble_archived_sources_exactly() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for split in manifest["splits"]:
        archived_source = _read_repo_path(split["source_archive_path"])
        source = archived_source[split["source_start_byte"] :]
        assert _sha256(source) == split["source_sha256"]
        chunks = [_read_repo_path(chunk_path) for chunk_path in split["chunks"]]
        assert all(len(chunk) <= MAX_CHUNK_BYTES for chunk in chunks)
        assert b"".join(chunks) == source


def test_current_handoff_is_a_short_index_and_local_links_resolve() -> None:
    handoff = ROOT / "AGENT_HANDOFF.md"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    handoff_bytes = handoff.read_bytes()
    assert len(handoff_bytes.splitlines()) < 150
    root_snapshot = next(
        item
        for item in manifest["snapshots"]
        if item["original_path"] == "AGENT_HANDOFF.md"
    )
    original_index = _read_repo_path(root_snapshot["archive_path"])
    archived_prefix = original_index[: manifest["root_index_prefix_bytes"]]
    assert _sha256(archived_prefix) == manifest["root_index_prefix_sha256"]
    assert b"## Active Study and Current Status" in handoff_bytes
    assert b"## Peer Record Index" in handoff_bytes
    assert b"does not independently validate" in handoff_bytes

    markdown_paths = [handoff, *(ROOT / p for p in manifest["navigation_documents"])]
    for markdown_path in markdown_paths:
        contents = markdown_path.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(contents):
            target = urlsplit(match.group(1).strip("<>"))
            if target.scheme or target.netloc or not target.path:
                continue
            local_path = (markdown_path.parent / unquote(target.path)).resolve()
            assert local_path.is_relative_to(ROOT.resolve()), (
                f"link escapes repository: {match.group(1)}"
            )
            assert local_path.exists(), (
                f"broken local link in {markdown_path.relative_to(ROOT)}: {match.group(1)}"
            )


pytestmark = pytest.mark.unit
