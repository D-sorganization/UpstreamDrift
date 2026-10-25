"""Keep the hand audit's model rows synchronized with the capability inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "src" / "config" / "grip_capability_inventory.json"
AUDIT = ROOT / "docs" / "development" / "full_body_models" / "HANDS_AUDIT.md"


def test_hands_audit_model_rows_match_grip_capability_inventory() -> None:
    """Every inventory id appears once as a model row in the hands audit."""
    models = json.loads(INVENTORY.read_text(encoding="utf-8"))["models"]
    expected_ids = {model["id"] for model in models}
    assert len(expected_ids) == len(models), "inventory model ids must be unique"

    audit_text = AUDIT.read_text(encoding="utf-8")
    matrix = audit_text.split("## Per-Asset Hand Capability Matrix", maxsplit=1)[1]
    matrix = matrix.split("\n## ", maxsplit=1)[0]
    row_ids = re.findall(r"^\|\s*`([^`]+)`\s*\|", matrix, re.MULTILINE)
    assert len(row_ids) == len(set(row_ids)), "audit model rows must be unique"
    assert set(row_ids) == expected_ids
