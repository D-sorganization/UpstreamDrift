"""Keep neural motion matching prose consistent with diagnostic receipts (#13778)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN_DIR = REPO_ROOT / "docs" / "plans" / "neural_motion_matching"


@pytest.mark.parametrize(
    ("receipt_name", "document_name", "empty_path", "claim_pattern"),
    [
        (
            "nm12_model_cards_turnover_receipt.json",
            "model_cards_reproduction_turnover.md",
            ("card_catalog", "promoted_models"),
            r"\bpromoted\b|\bpromotion verdict\b|\bqualified\b",
        ),
        (
            "nm09_checkpoint_matrix_receipt.json",
            "checkpoint_matrix.md",
            ("roster_coverage", "qualified_native"),
            r"\bqualified native\b|\bnative replay verified\b",
        ),
    ],
)
def test_empty_diagnostic_receipt_has_no_positive_document_claim(
    receipt_name: str,
    document_name: str,
    empty_path: tuple[str, str],
    claim_pattern: str,
) -> None:
    """An empty receipt roster cannot accompany positive promotion/qualification prose."""
    receipt_path = PLAN_DIR / "evidence" / receipt_name
    document_path = PLAN_DIR / document_name
    assert receipt_path.is_file(), f"Missing receipt: {receipt_path}"
    assert document_path.is_file(), f"Missing document: {document_path}"
    assert len(empty_path) == 2, (
        "Receipt field path must identify one nested roster field"
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt.get("status") == "DIAGNOSTIC"
    roster = receipt
    for key in empty_path:
        assert isinstance(roster, dict), f"Invalid receipt path: {empty_path}"
        roster = roster.get(key)
    assert roster == [], f"Expected empty receipt field {empty_path}: {roster!r}"

    text = document_path.read_text(encoding="utf-8")
    assert receipt_name in text, f"{document_name} must link its evidence receipt"
    matches = [
        line.strip()
        for line in text.splitlines()
        if re.search(claim_pattern, line, re.IGNORECASE)
        and "diagnostic" not in line.lower()
        and "unqualified" not in line.lower()
        and not re.search(r"\b(no|not|never|cannot|does not)\b", line, re.IGNORECASE)
    ]
    assert not matches, (
        f"{document_name} makes positive claims while {receipt_name} has "
        f"{empty_path[-1]}=[]: {matches}"
    )
