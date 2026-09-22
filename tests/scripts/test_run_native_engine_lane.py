"""Unit tests for native engine lane receipts (MS-43 #10342)."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from scripts.ci import run_native_engine_lane as lane

pytestmark = pytest.mark.unit


def _valid_receipt(*, status: str = "pass", executed: int = 3) -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "engine": "opensim",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "checker_source_sha256": "a" * 64,
        "contract_hashes": dict.fromkeys(lane.CONTRACT_PATHS, "b" * 64),
        "repository_revision": "c" * 40,
        "source_freshness": {"status": "clean"},
        "engine_inventory": {
            "available": True,
            "version": "4.6.0",
            "source_sha256": "d" * 64,
        },
        "tests": {
            "collected": executed,
            "passed": executed,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "executed": executed,
        },
        "pytest_probe": {
            "name": "native_pytest_lane",
            "status": "pass" if status == "pass" else "fail",
            "returncode": 0 if status == "pass" else 1,
            "timed_out": False,
        },
    }


def test_validate_receipt_accepts_well_formed_pass_receipt() -> None:
    ok, reasons = lane.validate_receipt(_valid_receipt())
    assert ok is True
    assert reasons == []


def test_validate_receipt_fails_closed_on_missing_contract_hashes() -> None:
    receipt = _valid_receipt()
    receipt["contract_hashes"] = {}
    ok, reasons = lane.validate_receipt(receipt)
    assert ok is False
    assert any("contract" in reason for reason in reasons)


def test_validate_receipt_rejects_pass_with_zero_executed_tests() -> None:
    receipt = _valid_receipt(status="pass", executed=0)
    ok, reasons = lane.validate_receipt(receipt)
    assert ok is False
    assert any("nonzero executed" in reason for reason in reasons)


def test_assess_receipt_freshness_ok_warn_fail_thresholds() -> None:
    now = datetime(2026, 9, 21, tzinfo=UTC)
    fresh = (now - timedelta(days=3)).isoformat()
    stale = (now - timedelta(days=10)).isoformat()
    expired = (now - timedelta(days=31)).isoformat()

    assert lane.assess_receipt_freshness(fresh, now=now) == ("ok", pytest.approx(3.0))
    assert lane.assess_receipt_freshness(stale, now=now) == (
        "warn",
        pytest.approx(10.0),
    )
    assert lane.assess_receipt_freshness(expired, now=now) == (
        "fail",
        pytest.approx(31.0),
    )


def test_contract_hashes_include_lane_contract_files() -> None:
    hashes = lane.contract_hashes()
    for path in lane.CONTRACT_PATHS:
        assert path in hashes
        assert len(hashes[path]) == 64


def test_parse_junit_extracts_case_statuses(tmp_path) -> None:
    junit = tmp_path / "lane.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite tests="2" failures="1" errors="0" skipped="0">
  <testcase classname="tests.opensim.test_a" name="test_pass"/>
  <testcase classname="tests.opensim.test_b" name="test_fail">
    <failure message="boom"/>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )
    parsed = lane._parse_junit(junit)
    assert parsed["collected"] == 2
    assert parsed["passed"] == 1
    assert parsed["failed"] == 1
    assert parsed["cases"][1]["status"] == "fail"


def test_build_receipt_marks_fail_when_probe_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        lane,
        "_engine_inventory",
        lambda **kwargs: {
            "available": False,
            "version": None,
            "source_sha256": None,
            "error": "missing",
        },
    )
    monkeypatch.setattr(
        lane,
        "run_pytest_lane",
        lambda **kwargs: {
            "name": "native_pytest_lane",
            "status": "fail",
            "returncode": 5,
            "timed_out": False,
            "result": {
                "collected": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "cases": [],
            },
        },
    )
    receipt = lane.build_receipt(
        engine="opensim",
        out_dir=lane.DEFAULT_OUT_DIR,
        python="python",
        generated_at=datetime(2026, 9, 21, tzinfo=UTC),
    )
    assert receipt["status"] == "fail"
    assert receipt["tests"]["executed"] == 0
    ok, _ = lane.validate_receipt(receipt)
    assert ok is True


def test_validate_only_cli(tmp_path, capsys) -> None:
    receipt_path = tmp_path / "opensim_receipt.json"
    receipt_path.write_text(
        __import__("json").dumps(_valid_receipt()),
        encoding="utf-8",
    )
    code = lane.main(["--engine", "opensim", "--validate-only", str(receipt_path)])
    assert code == 0
    assert "Valid:" in capsys.readouterr().out
