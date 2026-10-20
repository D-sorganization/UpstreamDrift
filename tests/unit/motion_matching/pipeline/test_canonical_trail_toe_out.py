"""The committed canonical receipts carry the trail-foot toe-out (R3, #12683).

R3 found that canonical runs never enabled foot progression, so the trail
(right) foot stood square at address. PR #12731 made ``style`` the canonical
mode and PR #12814 (R3b, #12789) added the static tibial-torsion offset that
lets the 7-iron reach the target. These tests pin the outcome on the committed
receipts, not only on the code paths, so a regeneration that drops the mode or
loses the trail toe-out fails here.

Target: ``TRAIL_TOE_OUT_STYLE_DEG`` = 20 deg, the OSV-4 documented per-foot
toe-out (``DEFAULT_TOE_OUT_DEG``, #11730) adopted by owner decision B on #12678
(2026-10-15, "about 20 deg"). It is a style target, so the receipt must tag it
``style``, never ``capture``. Acceptance: each foot within 2 deg of its adopted
target (the OSV-4 acceptance tolerance; the issue's acceptance checklist).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.foot_progression import (
    TRAIL_TOE_OUT_STYLE_DEG,
)
from src.shared.python.motion_matching.pipeline.address_feet import (
    CANONICAL_FOOT_PROGRESSION,
)

pytestmark = pytest.mark.unit

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)
#: Canonical static-seed receipts (driver, 7-iron), right-handed golfer.
CANONICAL = ("anthro_driver_seeds", "anthro_iron_seeds_zmp")
#: OSV-4 (#11730) acceptance: model foot within 2 deg of its adopted target.
ACCEPTANCE_DEG = 2.0
#: Owner style toe-out for the trail foot, in degrees (OSV-4 / decision B).
EXPECTED_TRAIL_TOE_OUT_DEG = 20.0


def _receipt(name: str) -> dict:
    return json.loads((EVIDENCE / name / "receipt.json").read_text(encoding="utf-8"))


def _feet(name: str) -> dict:
    block = _receipt(name)["address"]["foot_progression"]
    assert block["enabled"] is True, f"{name}: foot progression disabled"
    return block["feet"]


def test_style_constant_is_the_cited_twenty_degrees() -> None:
    assert CANONICAL_FOOT_PROGRESSION == "style"
    assert pytest.approx(EXPECTED_TRAIL_TOE_OUT_DEG) == TRAIL_TOE_OUT_STYLE_DEG


@pytest.mark.parametrize("name", CANONICAL)
def test_canonical_receipt_ran_with_style_foot_progression(name: str) -> None:
    receipt = _receipt(name)
    assert receipt["foot_progression_mode"] == CANONICAL_FOOT_PROGRESSION
    assert receipt["address"]["foot_progression"]["source"] == "style"
    assert set(receipt["foot_seed_deg"]) == {"hip_rotation_l", "hip_rotation_r"}


@pytest.mark.parametrize("name", CANONICAL)
def test_trail_foot_toes_out_to_the_style_target(name: str) -> None:
    trail = _feet(name)["right"]
    assert trail["role"] == "trail"
    assert trail["target_source"] == "style"
    assert trail["target_deg"] == pytest.approx(EXPECTED_TRAIL_TOE_OUT_DEG)
    assert abs(trail["model_deg"] - EXPECTED_TRAIL_TOE_OUT_DEG) <= ACCEPTANCE_DEG, (
        f"{name}: trail toe-out {trail['model_deg']:.2f} deg is not within "
        f"{ACCEPTANCE_DEG} deg of {EXPECTED_TRAIL_TOE_OUT_DEG} deg"
    )


@pytest.mark.parametrize("name", CANONICAL)
def test_lead_foot_keeps_its_capture_target(name: str) -> None:
    lead = _feet(name)["left"]
    assert lead["role"] == "lead"
    assert lead["target_source"] == "capture"
    assert abs(lead["model_deg"] - lead["target_deg"]) <= ACCEPTANCE_DEG
