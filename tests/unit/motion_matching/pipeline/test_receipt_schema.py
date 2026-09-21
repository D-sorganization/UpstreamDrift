"""Tests for ground-support execution receipt schema and validator (HO-2 #10156)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.pipeline.receipt_schema import (
    Receipt,
    validate_receipt,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
RECEIPTS_ROOT = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "evidence"
    / "ground_support"
)
DRIVER_RECEIPT_PATH = RECEIPTS_ROOT / "anthro_driver" / "receipt.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file contents as a dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_driver_receipt_validates_and_has_expected_zmp() -> None:
    """Step 1 (TDD): Committed driver receipt validates and has expected ZMP."""
    doc = load_json(DRIVER_RECEIPT_PATH)
    receipt = validate_receipt(doc)
    assert isinstance(receipt, Receipt)
    assert receipt.dynamics.reference_zmp is not None
    assert receipt.dynamics.reference_zmp.outside_fraction_1s_to_1_5s == pytest.approx(
        0.6666666666666666, rel=1e-5
    )


def test_all_committed_receipts_validate() -> None:
    """All committed receipt.json files under evidence/ground_support/** validate."""
    receipt_files = sorted(RECEIPTS_ROOT.glob("**/receipt.json"))
    assert len(receipt_files) >= 10, (
        f"Expected at least 10 receipts, found {len(receipt_files)}"
    )

    for receipt_path in receipt_files:
        doc = load_json(receipt_path)
        receipt = validate_receipt(doc)
        assert isinstance(receipt, Receipt), f"Failed to validate {receipt_path}"
        assert receipt.capture is None or isinstance(receipt.capture, str)


def test_rejects_missing_dynamics_reference_zmp() -> None:
    """Negative test 1: Reject receipt with missing dynamics.reference_zmp."""
    doc = load_json(DRIVER_RECEIPT_PATH)
    del doc["dynamics"]["reference_zmp"]
    with pytest.raises(ValueError, match=r"dynamics\.reference_zmp"):
        validate_receipt(doc)


def test_rejects_negative_marker_rms() -> None:
    """Negative test 2: Reject receipt with negative marker_rms_m."""
    doc = load_json(DRIVER_RECEIPT_PATH)
    doc["ik"]["marker_rms_m"] = -0.025
    with pytest.raises(ValueError, match=r"ik\.marker_rms_m"):
        validate_receipt(doc)


def test_rejects_missing_required_section() -> None:
    """Negative test 3: Reject receipt with missing required top-level section."""
    doc = load_json(DRIVER_RECEIPT_PATH)
    del doc["dynamics"]
    with pytest.raises(ValueError, match=r"dynamics"):
        validate_receipt(doc)
