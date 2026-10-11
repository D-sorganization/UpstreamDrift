"""Keep imported peer handoffs intact while satisfying document size limits."""

import hashlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
HANDOFFS = ROOT / "docs/agents/handoffs"


def test_peer_record_parts_preserve_original_bytes() -> None:
    parts = sorted(HANDOFFS.glob("shot-pattern-integration-peer-records-part-*.md"))
    assert len(parts) >= 3
    assert all(part.stat().st_size <= 50 * 1024 for part in parts)
    restored = b"\n".join(part.read_bytes() for part in parts)
    assert hashlib.sha256(restored).hexdigest() == (
        "bf292cd335b83d70a52c91ead19ff27781a857fe2d7f2513f8f5bbb5e243b564"
    )


def test_peer_record_index_links_every_preserved_section() -> None:
    index = (HANDOFFS / "shot-pattern-integration-peer-records.md").read_text()
    for part in sorted(
        HANDOFFS.glob("shot-pattern-integration-peer-records-part-*.md")
    ):
        assert part.name in index
        for line in part.read_text().splitlines():
            if line.startswith("# "):
                assert line in index
