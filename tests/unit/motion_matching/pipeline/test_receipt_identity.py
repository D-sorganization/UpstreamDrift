"""Canonical receipt byte identity (#14771)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.shared.python.motion_matching.receipt_identity import (
    canonical_receipt_bytes,
    receipts_canonical_bytes_equal,
)

pytestmark = pytest.mark.unit


def test_canonical_bytes_stable_under_key_order() -> None:
    a = {"z": 1, "a": {"m": 2, "b": 3}}
    b = {"a": {"b": 3, "m": 2}, "z": 1}
    assert canonical_receipt_bytes(a) == canonical_receipt_bytes(b)


def test_receipts_canonical_bytes_equal_files() -> None:
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "r1.json"
        p2 = Path(td) / "r2.json"
        obj = {"x": [1, 2], "y": {"k": None}}
        p1.write_text(json.dumps(obj), encoding="utf-8")
        p2.write_text(json.dumps({"y": {"k": None}, "x": [1, 2]}), encoding="utf-8")
        assert receipts_canonical_bytes_equal(p1, p2)


def test_receipts_canonical_bytes_equal_rejects_different_values() -> None:
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "r1.json"
        p2 = Path(td) / "r2.json"
        p1.write_text('{"a":1}', encoding="utf-8")
        p2.write_text('{"a":2}', encoding="utf-8")
        assert not receipts_canonical_bytes_equal(p1, p2)
