"""Tests for per-package coverage aggregation and ratchet evaluation (#14946)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any
import pytest

from scripts.ci.package_coverage_ratchet import (
    check_coverage_freshness,
    compute_package_coverage,
    evaluate_package_coverage_ratchet,
    main,
    ratchet_from_measured,
)

pytestmark = pytest.mark.unit


def _s(c: int, s: int) -> dict[str, Any]:
    return {
        "summary": {
            "covered_lines": c,
            "num_statements": s,
            "percent_covered": 100.0 * c / s,
        }
    }


def _g(n: str, p: str, m: float) -> dict[str, Any]:
    return {"name": n, "path": p, "min_coverage": m}


@pytest.fixture
def sample_data() -> tuple[dict[str, Any], dict[str, Any]]:
    cov = {
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
        "files": {
            "src/shared/python/data_io/loader.py": _s(30, 50),
            "src/shared/python/data_io/saver.py": _s(25, 50),
            "src/api/routes/health.py": _s(90, 100),
        },
    }
    b = {
        "coverage_gates": [
            _g("data-io", "src/shared/python/data_io/", 50.0),
            _g("api-routes", "src/api/routes/", 85.0),
        ]
    }
    return cov, b


def test_compute_package_coverage(
    sample_data: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    cov, b = sample_data
    sums = compute_package_coverage(cov, package_gates=b["coverage_gates"])
    assert sums["data-io"].covered_lines == 55 and sums["data-io"].num_statements == 100
    assert (
        sums["data-io"].percent_covered == 55.0 and sums["data-io"].matching_files == 2
    )
    assert (
        sums["api-routes"].covered_lines == 90
        and sums["api-routes"].percent_covered == 90.0
    )


def test_evaluate_package_coverage_ratchet(
    sample_data: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    cov, b = sample_data
    assert evaluate_package_coverage_ratchet(cov, b).passed is True
    b_fail = {"coverage_gates": [_g("data-io", "src/shared/python/data_io/", 60.0)]}
    rep_fail = evaluate_package_coverage_ratchet(cov, b_fail)
    assert rep_fail.passed is False and "dropped below" in rep_fail.results[0].reason
    b_miss = {"coverage_gates": [_g("missing", "src/none/", 50.0)]}
    assert evaluate_package_coverage_ratchet(cov, b_miss).passed is False


def test_ratchet_from_measured(
    sample_data: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    cov, b = sample_data
    r = ratchet_from_measured(cov, b, ratchet_up_only=True)
    assert r["data-io"] == 55.0 and r["api-routes"] == 90.0
    high_b = {"coverage_gates": [_g("data-io", "src/shared/python/data_io/", 75.0)]}
    assert ratchet_from_measured(cov, high_b, ratchet_up_only=True)["data-io"] == 75.0


def test_check_coverage_freshness() -> None:
    now = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    fresh, msg = check_coverage_freshness(
        {"meta": {"timestamp": (now - timedelta(days=2)).isoformat()}},
        max_age_days=7.0,
        now=now,
    )
    assert fresh is True and "fresh" in msg
    stale, msg = check_coverage_freshness(
        {"meta": {"timestamp": (now - timedelta(days=8)).isoformat()}},
        max_age_days=7.0,
        now=now,
    )
    assert stale is False and "stale" in msg
    assert check_coverage_freshness({"meta": {}})[0] is False


def test_cli_end_to_end(
    tmp_path: Path, sample_data: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    cov, b = sample_data
    c_f, b_f, s_f = tmp_path / "c.json", tmp_path / "b.json", tmp_path / "s.md"
    c_f.write_text(json.dumps(cov)), b_f.write_text(json.dumps(b))
    cmd = [
        "--coverage-json",
        str(c_f),
        "--budget-json",
        str(b_f),
        "--check-freshness",
        "--summary-out",
        str(s_f),
    ]
    assert main(cmd) == 0 and s_f.is_file() and "Passed" in s_f.read_text()
    b["coverage_gates"][0]["min_coverage"] = 99.0
    b_f.write_text(json.dumps(b))
    assert main(["--coverage-json", str(c_f), "--budget-json", str(b_f)]) == 1
