"""Tests for the model-matching status report (MATCH-1 #14443).

Enforces:
1. docs/development/model_matching/model_matching_status.json has one complete entry per engine.
2. Unverified numbers and unmeasured metrics are marked in the rendered table.
3. The generated table in docs/development/model_matching/STATUS.md matches the
   rendering from the ledger byte-for-byte.
4. The ledger and report carry no private host details or raw capture paths.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from scripts.generate_model_matching_status import (
    FENCE_END,
    FENCE_START,
    REQUIRED_ENGINES,
    REQUIRED_FIELDS,
    render_model_matching_status,
    validate_ledger,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = (
    REPO_ROOT / "docs" / "development" / "model_matching" / "model_matching_status.json"
)
STATUS_PATH = REPO_ROOT / "docs" / "development" / "model_matching" / "STATUS.md"

PRIVATE_PATTERNS = (
    re.compile(r"/home/"),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    re.compile(r"\b(?:brick|controltower|tailnet|ts\.net)\b", re.IGNORECASE),
    re.compile(r"fleet24"),
    re.compile(r"\.c3d\b", re.IGNORECASE),
)


def _ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _entry(**overrides: object) -> dict:
    entry = {
        "engine": "mujoco",
        "model": "m",
        "approach": "a",
        "key_parameters": "p",
        "marker_rms_mm": 1.0,
        "joint_angle_rms_deg": None,
        "verified": True,
        "evidence": "e",
        "date": "2026-10-10",
        "verdict": "REJECTED",
        "blocker": "b",
    }
    entry.update(overrides)
    return entry


def test_ledger_covers_every_engine_with_required_fields() -> None:
    ledger = _ledger()
    engines = [entry["engine"] for entry in ledger["engines"]]
    assert set(REQUIRED_ENGINES) <= set(engines)
    for entry in ledger["engines"]:
        assert set(REQUIRED_FIELDS) <= set(entry), entry["engine"]
    validate_ledger(ledger)


def test_ledger_never_claims_public_readiness() -> None:
    assert _ledger()["public_ready"] is False


def test_unverified_and_unmeasured_values_are_marked() -> None:
    ledger = {
        "as_of": "2026-11-04",
        "public_ready": False,
        "engines": [
            _entry(engine="drake", verified=False, marker_rms_mm=None),
        ],
    }
    table = render_model_matching_status(ledger)
    assert "UNVERIFIED" in table
    assert "not measured" in table


def test_validate_ledger_rejects_missing_field_and_bad_date() -> None:
    entry = _entry()
    del entry["blocker"]
    with pytest.raises(ValueError, match="blocker"):
        validate_ledger(
            {"as_of": "2026-11-04", "public_ready": False, "engines": [entry]}
        )
    with pytest.raises(ValueError, match="date"):
        validate_ledger(
            {
                "as_of": "2026-11-04",
                "public_ready": False,
                "engines": [_entry(date="10/10")],
            }
        )


def test_status_doc_table_matches_ledger() -> None:
    text = STATUS_PATH.read_text(encoding="utf-8")
    start = text.index(FENCE_START) + len(FENCE_START)
    end = text.index(FENCE_END)
    assert text[start:end].strip() == render_model_matching_status(_ledger()).strip()


@pytest.mark.parametrize("path", [LEDGER_PATH, STATUS_PATH], ids=["ledger", "status"])
def test_no_private_host_details(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in PRIVATE_PATTERNS:
        assert not pattern.search(text), f"{path.name} matches {pattern.pattern}"
