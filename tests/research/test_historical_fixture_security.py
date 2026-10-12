"""Fail Closed Before Executing Reviewed Historical Test Fixtures."""

from pathlib import Path

import pytest

from tests.research import historical_fixture_security as fixtures

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("key", ["article", "geometry", "renderer", "review"])
@pytest.mark.parametrize("archived", [False, True])
def test_reviewed_fixture_loads_and_rejects_changed_bytes(monkeypatch, key, archived):
    original = fixtures.read_fixture(key, archived=archived)
    assert original
    read = Path.read_bytes

    def changed(path):
        data = read(path)
        return data + b"# unreviewed change\n"

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="digest"):
        fixtures.read_fixture(key, archived=archived)


def test_unknown_fixture_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        fixtures.read_fixture("../../untrusted")


def test_redirected_fixture_rejected_before_read(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda self: Path("/untrusted/fixture.py"))
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail("read escaped path")
    )
    with pytest.raises(ValueError, match="path"):
        fixtures.read_fixture("article")


def test_unreviewed_source_rejected_before_ast_execution():
    with pytest.raises(ValueError, match="digest"):
        fixtures.verify_source("article", "raise RuntimeError('untrusted')")


def test_archive_raw_digest_checked_independently(monkeypatch):
    monkeypatch.setattr(fixtures.gzip, "decompress", lambda _: b"unreviewed raw source")
    with pytest.raises(ValueError, match="digest"):
        fixtures.read_fixture("article", archived=True)
