"""#12030: the nominal (uncalibrated) receipts' trajectory IK stays in the
address basin.

The 7-iron anthropometric document carries the driver's head-triad offsets, so
before the triad-shape gate the face-orientation residual targeted a face
rotated 40-80 degrees and dragged the nominal 7-iron IK to 265 mm RMS. These
checks read the committed receipts, so a regeneration that regresses fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/ground_support"
NOMINAL_RUNS = ("anthro_driver", "anthro_iron")
#: Issue #12030 acceptance: whole-run and frame-0 nominal IK marker RMS.
MAX_NOMINAL_IK_RMS_M = 0.080
#: Issue #12030 acceptance: frame 0 may not exceed the address residual by more.
MAX_FRAME0_JUMP_M = 0.050


def _receipt(run: str) -> dict:
    return json.loads((EVIDENCE / run / "receipt.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("run", NOMINAL_RUNS)
def test_nominal_frame0_ik_stays_at_the_address(run: str) -> None:
    receipt = _receipt(run)
    address = receipt["address"]["calibrated"]["marker_rms_m"]
    frame0 = receipt["ik"]["frame0_marker_rms_m"]
    assert frame0 - address <= MAX_FRAME0_JUMP_M, (frame0, address)
    assert frame0 <= MAX_NOMINAL_IK_RMS_M


@pytest.mark.parametrize("run", NOMINAL_RUNS)
def test_nominal_ik_is_bounded_without_range_flags(run: str) -> None:
    ik = _receipt(run)["ik"]
    assert ik["marker_rms_m"] <= MAX_NOMINAL_IK_RMS_M
    assert ik["range_of_motion_flags"] == {}


def test_nominal_iron_records_why_the_face_residual_is_off() -> None:
    face = _receipt("anthro_iron")["ik"]["face_orientation"]
    assert face["available"] is False
    assert "do not match the captured triad" in face["reason"]
