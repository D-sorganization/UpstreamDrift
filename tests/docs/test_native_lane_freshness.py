"""Freshness gate for OpenSim/MyoSuite nightly native lane receipts (MS-43 #10342).

Warns when a committed receipt is older than seven days; fails closed at thirty
days so stale native-lane evidence cannot masquerade as current on ``main``.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.ci.run_native_engine_lane import (
    DEFAULT_OUT_DIR,
    ENGINE_LANES,
    assess_receipt_freshness,
    validate_receipt,
)

pytestmark = pytest.mark.unit

EVIDENCE_DIR = DEFAULT_OUT_DIR


@pytest.mark.parametrize("engine", tuple(ENGINE_LANES))
def test_native_lane_receipt_is_committed(engine: str) -> None:
    receipt_path = EVIDENCE_DIR / ENGINE_LANES[engine]["receipt_filename"]
    assert receipt_path.is_file(), (
        f"Missing native lane receipt at {receipt_path}. "
        "Run `bash scripts/ci/run_native_engine_lane.sh --engine "
        f"{engine} --out {DEFAULT_OUT_DIR}` on ControlTower."
    )


@pytest.mark.parametrize("engine", tuple(ENGINE_LANES))
def test_native_lane_receipt_schema(engine: str) -> None:
    receipt_path = EVIDENCE_DIR / ENGINE_LANES[engine]["receipt_filename"]
    if not receipt_path.is_file():
        pytest.skip("receipt not yet committed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    ok, reasons = validate_receipt(receipt)
    assert ok, f"{receipt_path} failed validation: {reasons}"
    assert receipt.get("engine") == engine


@pytest.mark.parametrize("engine", tuple(ENGINE_LANES))
def test_native_lane_receipt_freshness(engine: str) -> None:
    receipt_path = EVIDENCE_DIR / ENGINE_LANES[engine]["receipt_filename"]
    if not receipt_path.is_file():
        pytest.skip("receipt not yet committed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    generated_at = receipt["generated_at"]
    level, age_days = assess_receipt_freshness(
        generated_at,
        now=datetime.now(tz=UTC),
    )
    message = (
        f"{receipt_path.name} is {age_days:.1f} days old "
        f"(generated_at={generated_at}). Refresh on ControlTower via "
        f"`bash scripts/ci/run_native_engine_lane.sh --engine {engine}`."
    )
    if level == "fail":
        pytest.fail(message)
    if level == "warn":
        warnings.warn(message, UserWarning, stacklevel=1)


def test_native_lane_receipt_warns_after_seven_days() -> None:
    stale = (
        datetime.now(tz=UTC).replace(microsecond=0) - timedelta(days=8)
    ).isoformat()
    level, _ = assess_receipt_freshness(stale)
    assert level == "warn"


def test_native_lane_receipt_fails_after_thirty_days() -> None:
    expired = (
        datetime.now(tz=UTC).replace(microsecond=0) - timedelta(days=31)
    ).isoformat()
    level, _ = assess_receipt_freshness(expired)
    assert level == "fail"
