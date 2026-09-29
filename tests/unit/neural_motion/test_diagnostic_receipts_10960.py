"""Tests for #10960: diagnostic receipts do not assert retracted claims.

Acceptance criteria:
- Receipts with top-level status DIAGNOSTIC have no models in qualified_native or promoted_models.
- No nested status/outcome key equals "passed" except validation.outcome.
- all_issues_completed is not true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
NM09_RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "neural_motion_matching"
    / "evidence"
    / "nm09_checkpoint_matrix_receipt.json"
)
NM12_RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "neural_motion_matching"
    / "evidence"
    / "nm12_model_cards_turnover_receipt.json"
)


def _load_receipt(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"Receipt file not found: {path}"
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def _find_nested_status_outcome_keys(
    node: Any, current_path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    """Recursively collect nested (non-root) keys named 'status' or 'outcome'."""
    results: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = current_path + (key,)
            if len(path) > 1 and key in ("status", "outcome"):
                results.append((path, value))
            results.extend(_find_nested_status_outcome_keys(value, path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            results.extend(
                _find_nested_status_outcome_keys(item, current_path + (str(idx),))
            )
    return results


def test_nm09_diagnostic_receipt_does_not_assert_qualification() -> None:
    receipt = _load_receipt(NM09_RECEIPT_PATH)
    assert receipt.get("status") == "DIAGNOSTIC"

    roster = receipt.get("roster_coverage", {})
    assert roster.get("qualified_native") == []

    # Verify models classified under the real enum status string 'unqualified'
    unqualified = roster.get("unqualified", [])
    assert "driven_double_pendulum" in unqualified
    assert "driven_triple_pendulum" in unqualified
    assert "constrained_upper_body_golfer" in unqualified

    # Verify limitations[0] no longer claims replay was verified
    limitations: Sequence[str] = receipt.get("limitations", [])
    assert len(limitations) > 0
    assert "replay verified" not in limitations[0].lower()


def test_nm12_diagnostic_receipt_does_not_assert_promotion_or_completion() -> None:
    receipt = _load_receipt(NM12_RECEIPT_PATH)
    assert receipt.get("status") == "DIAGNOSTIC"

    catalog = receipt.get("card_catalog", {})
    assert catalog.get("promoted_models") == []

    # Verify models moved into unmeasured list matching PromotionVerdict.UNMEASURED
    unmeasured = catalog.get("unmeasured_models", [])
    assert "driven_double_pendulum" in unmeasured
    assert "driven_triple_pendulum" in unmeasured
    assert "constrained_upper_body_golfer" in unmeasured

    # Verify epic status all_issues_completed is false
    epic_status = receipt.get("epic_status", {})
    assert epic_status.get("all_issues_completed") is False


@pytest.mark.parametrize("path", [NM09_RECEIPT_PATH, NM12_RECEIPT_PATH])
def test_diagnostic_receipts_have_no_passed_nested_status_except_validation(
    path: Path,
) -> None:
    receipt = _load_receipt(path)
    if receipt.get("status") != "DIAGNOSTIC":
        pytest.skip(f"Receipt at {path.name} is not DIAGNOSTIC")

    # In any DIAGNOSTIC receipt, no qualified_native or promoted_models
    if "roster_coverage" in receipt:
        assert receipt["roster_coverage"].get("qualified_native") == []
    if "card_catalog" in receipt:
        assert receipt["card_catalog"].get("promoted_models") == []

    # Check that no nested status/outcome key equals "passed" except validation.outcome
    nested = _find_nested_status_outcome_keys(receipt)
    for key_path, val in nested:
        if key_path == ("validation", "outcome"):
            continue
        assert str(val).lower() != "passed", (
            f"Nested key at {'.'.join(key_path)} unexpectedly equals {val!r} in {path.name}"
        )
