"""NSX-04 verification status doc and registry inventory (#12553)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "src/config/loop_nullspace_registry.json"
STATUS_DOC = ROOT / "docs/research/loop_nullspace/NSX-04_verification_status.md"

_EXPECTED_TOTAL = 94
_EXPECTED_VERIFIED = 50
_EXPECTED_UNVERIFIED = 44


def _torque_capacity() -> list[dict]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return list(data["torque_capacity"])


def test_nsq04_status_doc_lists_current_inventory_counts() -> None:
    """#12553: status doc tracks live registry counts (not the stale issue snapshot)."""
    text = STATUS_DOC.read_text(encoding="utf-8")
    for needle in (
        f"| Total `torque_capacity` entries | {_EXPECTED_TOTAL} |",
        f"| `VERIFIED` | {_EXPECTED_VERIFIED} |",
        f"| `UNVERIFIED` (all `assumed: true`) | {_EXPECTED_UNVERIFIED} |",
    ):
        assert needle in text, f"missing inventory row: {needle!r}"
    assert re.search(r"#12553", text)


def test_torque_capacity_registry_inventory_matches_status_doc() -> None:
    entries = _torque_capacity()
    verified = [e for e in entries if e["verification_status"] == "VERIFIED"]
    unverified = [e for e in entries if e["verification_status"] != "VERIFIED"]

    assert len(entries) == _EXPECTED_TOTAL
    assert len(verified) == _EXPECTED_VERIFIED
    assert len(unverified) == _EXPECTED_UNVERIFIED

    for entry in verified:
        assert entry["source"].get("citation_id"), entry["joint_slot"]
        assert entry["assumed"] is False

    for entry in unverified:
        assert entry["source"].get("citation_id") is None, entry["joint_slot"]
        assert entry["assumed"] is True
