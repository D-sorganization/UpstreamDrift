"""Checksum golden table for HRG-GB3 builder outputs."""

from __future__ import annotations

import json
from typing import Any

from src.shared.python.version_info import get_repo_root

GOLDEN_RECEIPT_PATH = (
    get_repo_root()
    / "docs/development/full_body_models/evidence/grip_builder/grip_builder_goldens.json"
)


def load_golden_receipt() -> dict[str, Any]:
    if not GOLDEN_RECEIPT_PATH.is_file():
        raise FileNotFoundError(GOLDEN_RECEIPT_PATH)
    return json.loads(GOLDEN_RECEIPT_PATH.read_text(encoding="utf-8"))


def golden_case_keys() -> tuple[str, ...]:
    receipt = load_golden_receipt()
    return tuple(case["key"] for case in receipt["cases"])
