"""Capture-fit spec vs. neutral-grip showpiece spec (#14551, supervisor D22).

``full_body_spec_anthro_driver_capture.json`` is the driver document as it was
fitted to the tour capture, with the fitted (about 100 degree strong) hand
rotations. It is *fitted to the capture, not a grip recommendation*: every
capture-matching receipt (ground-support, cross-engine) chains to it. The
flagship ``full_body_spec_anthro_driver.json`` carries the 15 degree neutral
grip. These tests keep the two apart: a change to the showpiece grip must not
silently alter a capture receipt, and the capture spec must not drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.full_body_spec import canonical_sha256

pytestmark = pytest.mark.unit

DOCS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
CAPTURE_SPEC = DOCS / "full_body_spec_anthro_driver_capture.json"
SHOWPIECE_SPEC = DOCS / "full_body_spec_anthro_driver.json"
GROUND_SUPPORT = DOCS / "evidence/ground_support"

#: Canonical hash of the capture-fit spec: the document every capture receipt
#: below recorded as ``base_spec_sha256`` before the showpiece grip changed.
CAPTURE_SPEC_CANONICAL_SHA256 = (
    "ab724a31539dcfc956b4dd3f7b75ebe380aa9319a2c442323b650c736d07eb99"
)
CAPTURE_RECEIPTS = ("anthro_driver", "anthro_driver_seeds", "anthro_driver_drake")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_capture_spec_is_pinned() -> None:
    """The capture-fit spec may only change together with its receipts."""
    assert canonical_sha256(_load(CAPTURE_SPEC)) == CAPTURE_SPEC_CANONICAL_SHA256


@pytest.mark.parametrize("name", CAPTURE_RECEIPTS)
def test_capture_receipts_chain_to_the_capture_spec(name: str) -> None:
    receipt = _load(GROUND_SUPPORT / name / "receipt.json")
    assert receipt["base_spec_sha256"] == CAPTURE_SPEC_CANONICAL_SHA256


@pytest.mark.parametrize("name", CAPTURE_RECEIPTS)
def test_capture_receipts_do_not_chain_to_the_showpiece_spec(name: str) -> None:
    """Changing the showpiece grip must never silently re-base a receipt."""
    receipt = _load(GROUND_SUPPORT / name / "receipt.json")
    assert receipt["base_spec_sha256"] != canonical_sha256(_load(SHOWPIECE_SPEC))


def test_showpiece_grip_is_neutral_and_capture_grip_stays_fitted() -> None:
    capture, showpiece = _load(CAPTURE_SPEC), _load(SHOWPIECE_SPEC)
    assert (
        capture["subject"]["grip_rotation_deg"]
        != (showpiece["subject"]["grip_rotation_deg"])
    )
    # the neutral grip has distinct lead and trail rotations (option b15, #14751)
    lead, trail = showpiece["subject"]["grip_rotation_deg"].values()
    assert lead != trail
    assert [round(v, 2) for v in lead] == [-85.55, 43.37, -26.14]
    assert [round(v, 2) for v in trail] == [75.77, 13.57, -173.54]
    # the capture spec keeps the fitted, hand-specific rotations
    assert (
        capture["subject"]["grip_rotation_deg"]["L"]
        != (capture["subject"]["grip_rotation_deg"]["R"])
    )
