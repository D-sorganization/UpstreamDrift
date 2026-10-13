"""Reviewed Source Identity Bounds for Historical Parity Tests.

These fixed digests are review anchors, independent of the editable manifest.
Source changes require explicit review of these anchors, never auto-refresh.
Only repository-owned fixtures are accepted; no user-supplied code is executed.
This is an integrity boundary, not a Python sandbox or scientific approval.
"""

import gzip
import hashlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

BASE = (
    Path(__file__).resolve().parents[2]
    / "docs/research/shot_pattern_analysis/phase2/validation"
)
# Current source, compressed archive and decompressed source identities reviewed
# at integration commit 6ce959f227; historical snapshots remain byte-for-byte.
FIXTURES: dict[str, tuple[str, str, str, str, str]] = {
    "article": (
        "article_source_repair_v1/root/verify.py",
        "03-verify.py.gz",
        "3f20f21ef718c1ad1ac3eededef6de23be1b3c3959d18aa4f5e77c73b0833b96",
        "30c1720bdd31f8e5439629bc65157d1930649d9a4d4aa7922b60a08f5c6db7a4",
        "fe7eebecedf5e7f685c52b52325aa893f57c82dc7e93e6c48416451db3b4db53",
    ),
    "geometry": (
        "independent_geometry_derivatives.py",
        "05-independent_geometry_derivatives.py.gz",
        "dba193eb84c8cb5197ae1b2905015bae814edd9ecaf0eba3d4beb3fbc8281b98",
        "61d3133fe7ccf5f5f464b301d05c316ed07e50b29098364ae87f0360b8673508",
        "f688de8ab5eaa0cdd7db17986063ac618d9edcdaf610215f73b0284e89f4b023",
    ),
    "renderer": (
        "selected_renderer_representative_v1/qualify.py",
        "10-qualify.py.gz",
        "f7d6d0501cbd0a2a4ca0e8a12d06286c4904c7788a7da7fb2fdd01d55f5e2fbf",
        "35ec938412e996c59ba2f2e8e1d0f0f0a80fb98dfde2c1cdc14c9cf4bd29bd23",
        "109e1a28d6cce6122a0bd73573e81ef92faf27d0dc086d1b96caef2ca58754ec",
    ),
    "review": (
        "v6_output_review_harness_v1/review_batch.py",
        "15-review_batch.py.gz",
        "6bca82546ca682285f574b14d16ada5d5428450182148a3b2b1573cea5d221e2",
        "50ad9c1e27f2070828e89dd4029a9ded871637e000bf94482bedeadecd23e939",
        "36f0bcc867043148c7ec8e71a810e1a9460d0e56e009c4304d1716f3db897d87",
    ),
    "final_audit": (
        "independent_final_audit.py",
        "04-independent_final_audit.py.gz",
        "15eaba0f588261698a66880916ab97310ab8f313bafd741f715bb929e4232a2e",
        "3f0c4777b23597a21f3e68ccd9a5f873345a8034f919e50560280970d88950bf",
        "1f3a6c5d4207bd328d74c7dd36c3a3a89b048dc97fb505ea5b75b1f83a0db829",
    ),
    "pilot_audit": (
        "independent_pilot_audit.py",
        "06-independent_pilot_audit.py.gz",
        "84a39a344bf09bdf1b6c6c5e033de34d14a39a11039d5e254c63c8cc33b03123",
        "763aabe3a30a67961cb1bbd0115cf5223e4792af1b649d90a95e3d363987522d",
        "17360a206ed4a7d08c33332f865a1f54040e46743ed8e52e55ee803ac23dc8a6",
    ),
    "preflight": (
        "review_supplement_preflight.py",
        "09-review_supplement_preflight.py.gz",
        "d1eef8781ed1a88aaf35345d5aa20beeff1bf2ad2addd63232431a947534c4ec",
        "78d6ef97838730e51e9673848a74a3b932b002b13e06e75e1b97674e0384a65b",
        "e3b59694d6c56a94a3bd3acbe1dc11fb88de61f8933b92ea313a073ff5190283",
    ),
}


def _record(key: str) -> tuple[str, str, str, str, str]:
    if key not in FIXTURES:
        raise ValueError("Unknown reviewed fixture")
    return FIXTURES[key]


def _digest(raw: bytes, expected: str) -> None:
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Reviewed fixture digest mismatch; explicit review required")


def verify_source(key: str, source: str) -> None:
    """Reject unreviewed text before selecting or executing any AST nodes."""
    record = _record(key)
    if hashlib.sha256(source.encode()).hexdigest() not in (record[2], record[4]):
        raise ValueError("Reviewed source digest mismatch; explicit review required")


def read_fixture(key: str, *, archived: bool = False) -> str:
    """Read only an exact repository path and verify bytes before use."""
    source, archive, current_sha, storage_sha, original_sha = _record(key)
    path = BASE / (
        "refinement_v3/historical_source_snapshots/" + archive if archived else source
    )
    if path.resolve() != path or not path.is_relative_to(BASE):
        raise ValueError("Reviewed fixture path escaped or redirected")
    raw = path.read_bytes()
    _digest(raw, storage_sha if archived else current_sha)
    if archived:
        raw = gzip.decompress(raw)
        _digest(raw, original_sha)
    return raw.decode()


def fixture_key_for_source(path: Path) -> str:
    """Select an existing caller by its exact source path, never basename/glob."""
    for key, record in FIXTURES.items():
        if path == BASE / record[0]:
            # Check current source identity and path before using its archive.
            read_fixture(key)
            return key
    raise ValueError("Unknown reviewed fixture source path")


@pytest.mark.parametrize("key", list(FIXTURES))
@pytest.mark.parametrize("archived", [False, True])
def test_reviewed_fixture_loads_and_rejects_changed_bytes(monkeypatch, key, archived):
    original = read_fixture(key, archived=archived)
    assert original
    read = Path.read_bytes

    def changed(path):
        data = read(path)
        return data + b"# unreviewed change\n"

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="digest"):
        read_fixture(key, archived=archived)


def test_unknown_fixture_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        read_fixture("../../untrusted")


def test_redirected_fixture_rejected_before_read(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda self: Path("/untrusted/fixture.py"))
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail("read escaped path")
    )
    with pytest.raises(ValueError, match="path"):
        read_fixture("article")


def test_unreviewed_source_rejected_before_ast_execution():
    with pytest.raises(ValueError, match="digest"):
        verify_source("article", "raise RuntimeError('untrusted')")


def test_archive_raw_digest_checked_independently(monkeypatch):
    monkeypatch.setattr(gzip, "decompress", lambda _: b"unreviewed raw source")
    with pytest.raises(ValueError, match="digest"):
        read_fixture("article", archived=True)


@pytest.mark.parametrize("key", list(FIXTURES))
def test_exact_source_path_selects_its_own_reviewed_archive(key):
    path = BASE / FIXTURES[key][0]
    assert fixture_key_for_source(path) == key


def test_same_basename_outside_reviewed_path_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        fixture_key_for_source(Path("/untrusted/independent_final_audit.py"))
