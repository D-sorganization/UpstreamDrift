"""#12702: the MuJoCo-native smoke runs the document its bounds were measured on.

The smoke test's dynamics bound (0.15 m) is a statement about the canonical
driver run: ``evidence/ground_support/anthro_driver/receipt.json`` records
``base_spec_file = full_body_spec_anthro_driver.json`` (measured club cluster,
per-capture grip rotation, driver club record). The pipeline's *default* spec
is the native qualified v2 document, whose own committed receipt
(``evidence/ground_support/receipt.json``) sits at 0.174 m dynamics and 0.51 m
on the driver capture, so the smoke must name the canonical document.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
GROUND_SUPPORT = ROOT / "docs/development/full_body_models/evidence/ground_support"


def test_smoke_driver_uses_the_canonical_receipt_spec() -> None:
    from tests.integration.motion_matching import (
        test_ground_support_capture_smoke as smoke,
    )

    receipt = json.loads(
        (GROUND_SUPPORT / "anthro_driver/receipt.json").read_text(encoding="utf-8")
    )
    spec = smoke.CAPTURE_SPECS["driver"]
    assert spec.is_file()
    assert spec.name == receipt["base_spec_file"]
    assert receipt["dynamics"]["marker_rms_m"] < smoke.DYNAMICS_RMS_BOUND_M
    assert receipt["ik"]["marker_rms_m"] < smoke.IK_RMS_BOUND_M
