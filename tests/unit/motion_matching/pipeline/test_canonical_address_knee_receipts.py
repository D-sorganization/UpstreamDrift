"""The committed canonical receipts carry the R4 address knee flexion (#12684).

R4 (#12788) made ``capture`` the canonical knee mode: the address IK pulls each
knee to the capture's address median where the markers measure it reliably,
and to the owner's 15 +/- 5 deg prior elsewhere, never hyperextended. PR #13102
regenerated the canonical static-seed receipts on that code. These tests pin
the outcome on the committed receipts, so a regeneration that drops the knee
block, falls back to ``off`` or locks a knee straight fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.address_knee import CANONICAL_KNEE_FLEXION

pytestmark = pytest.mark.unit

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)
#: Canonical static-seed receipts (driver, 7-iron), right-handed golfer.
CANONICAL = ("anthro_driver_seeds", "anthro_iron_seeds_zmp")
#: Right-handed golfer: the left knee leads, the right knee trails.
ROLES = {"left": "lead", "right": "trail"}


def _knee_block(name: str) -> dict:
    path = EVIDENCE / name / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    block = receipt["address"].get("knee_flexion_deg")
    assert block is not None, f"{name}: receipt has no address.knee_flexion_deg"
    return block


@pytest.mark.parametrize("name", CANONICAL)
def test_canonical_receipt_ran_with_the_canonical_knee_mode(name: str) -> None:
    block = _knee_block(name)
    assert block["enabled"] is True
    assert block["mode"] == CANONICAL_KNEE_FLEXION


@pytest.mark.parametrize("name", CANONICAL)
@pytest.mark.parametrize("side", sorted(ROLES))
def test_canonical_address_knee_meets_its_capture_target(name: str, side: str) -> None:
    knee = _knee_block(name)["knees"][side]
    assert knee["role"] == ROLES[side]
    assert knee["target_source"] == "capture"
    assert knee["capture"]["reliable"] is True
    assert knee["hyperextended"] is False
    assert knee["model_deg"] > 0.0
    assert knee["within_band"] is True, (
        f"{name} {side}: model {knee['model_deg']:.2f} deg is outside the band "
        f"around target {knee['target_deg']:.2f} deg"
    )
